# rag_advanced.py
import os
import re
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Pre-retrieval
from langchain.retrievers.multi_query import MultiQueryRetriever

# Post-retrieval (compression)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Reranker (옵션: LLM 재랭크)
from pydantic import BaseModel, Field

from retriever_pgvector import build_pgvector_retriever
from db import CONN_STR

load_dotenv()

# ============ 0) 공통 LLM/임베딩 ============
LLM_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")

llm_fast = ChatOpenAI(model=LLM_MODEL, temperature=0)
llm_careful = ChatOpenAI(model=LLM_MODEL, temperature=0)  # rerank/압축용

# ============ 1) Query Rewrite (대화 맥락 → 독립 질의) ============
rewrite_prompt = PromptTemplate.from_template(
    """당신은 법률 QA 보조자입니다.
다음 대화 이력과 최종 사용자 질문을 참고하여, 문맥 의존 표현을 제거한 **독립적인 질문** 한 문장으로 다시 작성하세요.
가능하면 구체적 키워드(사건명/쟁점/조항/판시사항)를 보존하세요.

[대화 이력]
{history}

[사용자 질문]
{question}

[출력 형식]
- 한 줄의 독립 질의만 출력
"""
)
rewrite_chain: Runnable = rewrite_prompt | llm_fast | StrOutputParser()

def rewrite_query(inputs: Dict) -> Dict:
    question = inputs["question"]
    history = inputs.get("history", "")
    rewritten = rewrite_chain.invoke({"history": history, "question": question}).strip()
    return {"question": question, "rewritten": rewritten, "history": history}

# ============ 2) Query Routing (의도에 따라 retriever 선택) ============
# 간단한 규칙 기반 라우팅: 판례번호/조문번호/따옴표(정확매칭) 등은 키워드 지향
CASE_ID_RE = re.compile(r"\b(20\d{2})\s*도?\s*\d{1,6}\b")  # 예시: 2016도12345
ARTICLE_RE = re.compile(r"\b제?\d+조\b")

def route_strategy(q: str) -> str:
    qn = q.lower()
    if '"' in qn or "“" in qn or "”" in qn:
        return "keyword"
    if CASE_ID_RE.search(q) or ARTICLE_RE.search(q):
        return "keyword"
    # 기본은 의미 유사도(semantic)
    return "semantic"

# retriever 후보들 (pgvector는 둘 다 같은 소스지만 파라미터가 다름)
retriever_semantic = build_pgvector_retriever(
    connection_string=CONN_STR, search_type="mmr", k=5, fetch_k=40, embedding_model=EMBED_MODEL,
)
retriever_keyword_like = build_pgvector_retriever(
    connection_string=CONN_STR, search_type="similarity", k=8, embedding_model=EMBED_MODEL,
)

def route_retriever(inputs: Dict):
    q = inputs["rewritten"]
    route = route_strategy(q)
    return {"route": route, **inputs}

# ============ 3) Query Expansion (다각도 파라프레이즈) ============
# LangChain MultiQueryRetriever를 "semantic" 경로에서 활용
expansion_prompt = PromptTemplate.from_template(
    """아래 질문에 대해, 서로 다른 관점/표현으로 3개의 검색 질의를 만들어라.
법률/판례 용어를 한국어로 유지하고, 핵심 키워드를 다양화하라.
질문: {question}
-- 출력은 각 줄당 하나의 질의 --
"""
)
def build_multiquery_retriever(base_retriever):
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_fast,
        prompt=expansion_prompt,
    )

multi_semantic = build_multiquery_retriever(retriever_semantic)

# ============ 4) Query Transformation (KR/EN 변환, 동의어 추가 등) ============
# 간단: 한국어 원질의 + 재작성 질의 + EN 번역을 모두 후보로 사용
translate_prompt = PromptTemplate.from_template(
    "Translate the legal question into English precisely, keep case-law terms.\nQ: {q}\nEN:"
)
def expand_queries(inputs: Dict) -> Dict:
    q = inputs["question"]
    rewritten = inputs["rewritten"]
    en = (translate_prompt | llm_fast | StrOutputParser()).invoke({"q": rewritten}).strip()
    candidates = list(dict.fromkeys([q, rewritten, en]))  # 중복 제거, 순서 보존
    return {"query_candidates": candidates, **inputs}

# ============ 5) Retrieval (route별 실행) ============
def retrieve_docs(inputs: Dict) -> Dict:
    route = inputs["route"]
    rewritten = inputs["rewritten"]
    candidates = inputs["query_candidates"]

    # keyword 라우트: 확장 없이 keyword_like retriever로 단건 질의
    if route == "keyword":
        docs = retriever_keyword_like.invoke(rewritten)
        return {"docs": docs, **inputs}

    # semantic 라우트: MultiQuery(확장) + semantic retriever 사용
    # MultiQueryRetriever는 내부적으로 여러 쿼리로 union set을 구성
    docs = multi_semantic.invoke(rewritten)
    # 추가로 우리가 만든 candidates도 탐색: (선택) 커버리지 보강
    for cq in candidates:
        more = retriever_semantic.invoke(cq)
        docs.extend(more)

    # 중복 제거 (page_content+metadata 기준 간단 dedup)
    uniq, seen = [], set()
    for d in docs:
        key = (d.page_content, tuple(sorted(d.metadata.items())))
        if key not in seen:
            uniq.append(d)
            seen.add(key)
    return {"docs": uniq, **inputs}

# ============ 6) Post-Retrieval: Reranker (LLM-as-a-judge) ============
class RerankItem(BaseModel):
    score: float = Field(..., description="Higher is better")

rerank_prompt = PromptTemplate.from_template(
    """당신은 법률 검색 성능 평가자입니다.
주어진 '질문'과 '후보 문서'의 관련성을 0.0~1.0 사이 점수로 채점하세요.
점수만 JSON으로 반환하세요: {{ "score": 0.0 }}

[질문]
{question}

[후보 문서]
{doc}
"""
)
def llm_rerank(question: str, docs: List[Document], top_k: int = 8) -> List[Tuple[Document, float]]:
    pairs = []
    for d in docs:
        score_str = (rerank_prompt | llm_careful | StrOutputParser()).invoke(
            {"question": question, "doc": d.page_content[:1800]}  # 너무 길면 절단
        )
        # score 파싱
        import json
        try:
            s = json.loads(score_str).get("score", 0.0)
        except Exception:
            try:
                s = float(re.findall(r"[0-1]\.\d+|[01]", score_str)[0])
            except Exception:
                s = 0.0
        pairs.append((d, float(s)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

def apply_rerank(inputs: Dict, top_k: int = 8) -> Dict:
    q = inputs["rewritten"]
    docs = inputs["docs"]
    ranked = llm_rerank(q, docs, top_k=top_k)
    return {"docs_ranked": [d for d, s in ranked], "scores": [s for d, s in ranked], **inputs}

# ============ 7) Post-Retrieval: Compression (문맥 압축/추출) ============
# LLMChainExtractor로 각 문서를 '질문 관련 스팬'만 발췌
compressor = LLMChainExtractor.from_llm(llm_careful)
def apply_compression(inputs: Dict, k_final: int = 4) -> Dict:
     """
     컨텍스트 압축은 retriever 래핑 없이, 선택된 문서 리스트에
     LLMChainExtractor를 직접 적용합니다.
     """
     base_docs = inputs["docs_ranked"][:k_final]
     q = inputs["rewritten"]
     # LLMChainExtractor는 직접 compress_documents(docs, query)를 제공합니다.
     compressed = compressor.compress_documents(base_docs, q)
     return {"docs_compressed": compressed, **inputs}

# ============ 8) 최종 답변 생성 (Citation 포함) ============
answer_prompt = PromptTemplate.from_template(
    """너는 저작권법 판례 기반 어시스턴트다. 아래 컨텍스트만 사용해 사실에 근거해 한국어로 답해라.
필요하면 조문/판례명을 언급하되 과장하거나 추측하지 말고, 모르면 모른다고 말하라.

[질문]
{question}

[컨텍스트(압축 문서)]
{context}

요구사항:
- 간결한 요점 → 필요시 항목화
- 끝에 '출처'로 문서 제목/식별자(있으면) 목록화
"""
)
def render_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("doc_title") or meta.get("source") or meta.get("doc_id") or "unknown"
        parts.append(f"[{i}] {src}\n{d.page_content}")
    return "\n\n".join(parts)

def generate_answer(inputs: Dict) -> Dict:
    q = inputs["question"]
    ctx = render_context(inputs["docs_compressed"])
    answer = (answer_prompt | llm_careful | StrOutputParser()).invoke({"question": q, "context": ctx})
    return {"answer": answer, **inputs}

# ============ 9) LCEL 파이프라인 엔트리 ============
# 입력: {"question": "...", "history": "...(optional)"}
advanced_rag_chain: Runnable = (
    RunnablePassthrough.assign()                     # 그대로 전달
    | RunnableLambda(rewrite_query)                  # rewrite
    | RunnableLambda(route_retriever)                # routing
    | RunnableLambda(expand_queries)                 # KR/EN 병행 후보
    | RunnableLambda(retrieve_docs)                  # retrieval (route별)
    | RunnableLambda(apply_rerank)                   # reranker
    | RunnableLambda(apply_compression)              # compression
    | RunnableLambda(generate_answer)                # answer
)

def ask_advanced(question: str, history: str = "") -> Dict:
    return advanced_rag_chain.invoke({"question": question, "history": history})

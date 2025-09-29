# rag_advanced.py
"""
Advanced RAG (Slim) - Query Rewrite + (옵션) Multi-Query Expansion + LLM Rerank

환경변수(.env):
- CHAT_MODEL=gpt-4o-mini           # 답변/리라이트/랭크용 LLM
- EMBED_MODEL=text-embedding-3-small
- CHROMA_DIR=./chroma_db
- CHROMA_COLLECTION=copyright_chunks

- USE_MULTIQUERY=true|false        # 멀티쿼리 사용 여부 (기본 true)
- MULTIQUERY_NUM=3                 # 확장 쿼리 개수 (기본 3)
- RETRIEVER_SEARCH_TYPE=mmr        # mmr | similarity (기본 mmr)
- RETRIEVER_K=6                    # 1차 리트리버 반환 k (기본 6)
- RERANK_TOP_K=6                   # 재랭크 후 상위 k (기본 6)
- CTX_MAX_DOCS=4                   # 최종 컨텍스트 문서 수 (기본 4)
- CTX_MAX_CHARS=800                # 문서당 컨텍스트 절단 길이 (기본 800)
"""

import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

from langchain.retrievers.multi_query import MultiQueryRetriever

from pydantic import BaseModel, Field

from retriever_chroma import build_chroma_retriever

load_dotenv()

# =========================
# 0) 공통 설정/LLM
# =========================
LLM_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

USE_MULTIQUERY = os.getenv("USE_MULTIQUERY", "true").lower() == "true"
MULTIQUERY_NUM = int(os.getenv("MULTIQUERY_NUM", "3"))

RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "mmr") 
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "6"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "6"))

CTX_MAX_DOCS = int(os.getenv("CTX_MAX_DOCS", "4"))
CTX_MAX_CHARS = int(os.getenv("CTX_MAX_CHARS", "800"))

# LLM
llm_fast = ChatOpenAI(model=LLM_MODEL, temperature=0)
llm_careful = ChatOpenAI(model=LLM_MODEL, temperature=0)

# =========================
# 1) Query Rewrite (대화문맥 → 독립 질의 1문장)
# =========================
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

# =========================
# 2) Retriever (Chroma)
# =========================
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "copyright_chunks")

retriever = build_chroma_retriever(
    persist_directory=CHROMA_DIR,
    collection_name=CHROMA_COLLECTION,
    search_type=RETRIEVER_SEARCH_TYPE, # mmr
    k=RETRIEVER_K,
    embedding_model=EMBED_MODEL,
)

# =========================
# 3) (옵션) Multi-Query Expansion
# =========================
# ✅ 변수명을 MultiQueryRetriever 기본 기대값인 "question"으로 맞춤
expansion_prompt = PromptTemplate.from_template(
    """아래 한국어 법률 질문을 서로 다른 관점/용어/세부키워드로 {n}개의 검색 질의로 만들어라.
질문: {question}
-- 각 줄에 하나의 검색 질의만 출력 --
"""
)

def build_multiquery_retriever(base_retriever):
    prompt = expansion_prompt.partial(n=str(MULTIQUERY_NUM)) # 3
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_fast,
        prompt=prompt,  # 이 프롬프트는 {question} 변수를 사용
    )

mqr = build_multiquery_retriever(retriever) if USE_MULTIQUERY else None

def retrieve_docs(inputs: Dict) -> Dict:
    """USE_MULTIQUERY 설정에 따라 단일 질의 or 멀티쿼리 검색."""
    q = inputs["rewritten"]

    docs: List[Document] = []
    if USE_MULTIQUERY and mqr is not None:
        # 문자열로 바로 넘겨도 내부에서 {"question": ...}에 매핑됨
        docs = mqr.invoke(q)
    else:
        docs = retriever.invoke(q)

    # 중복 제거
    uniq, seen = [], set()
    for d in docs:
        key = (d.page_content, tuple(sorted((d.metadata or {}).items())))
        if key not in seen:
            uniq.append(d)
            seen.add(key)

    return {"docs": uniq, **inputs}

# =========================
# 4) Post: LLM Rerank
# =========================
class RerankItem(BaseModel):
    score: float = Field(..., description="Higher is better")

rerank_prompt = PromptTemplate.from_template(
    """당신은 법률 검색 성능 평가자입니다.
주어진 '질문'과 '후보 문서'의 관련성을 0.0~1.0 사이 점수로 채점하세요.
JSON으로 정확히 반환: {{ "score": 0.0 }}

[질문]
{question}

[후보 문서(발췌)]
{doc}
"""
)

def _score_one(question: str, doc_text: str) -> float:
    score_str = (rerank_prompt | llm_careful | StrOutputParser()).invoke(
        {"question": question, "doc": doc_text[:1800]}
    )
    import json, re
    try:
        return float(json.loads(score_str).get("score", 0.0))
    except Exception:
        try:
            m = re.findall(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", score_str)
            return float(m[0]) if m else 0.0
        except Exception:
            return 0.0

def llm_rerank(question: str, docs: List[Document], top_k: int) -> List[Tuple[Document, float]]:
    pairs: List[Tuple[Document, float]] = []
    for d in docs:
        s = _score_one(question, d.page_content or "")
        pairs.append((d, s))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

def apply_rerank(inputs: Dict) -> Dict:
    q = inputs["rewritten"]
    docs = inputs["docs"]
    ranked = llm_rerank(q, docs, top_k=RERANK_TOP_K)
    return {"docs_ranked": [d for d, s in ranked], "scores": [s for d, s in ranked], **inputs}

# =========================
# 5) 최종 답변 생성
# =========================
answer_prompt = PromptTemplate.from_template(
    """너는 저작권법 판례 기반 어시스턴트다. 아래 컨텍스트만 사용해 사실에 근거해 한국어로 답해라.
필요하면 조문/판례명을 언급하되 과장하거나 추측하지 말고, 모르면 모른다고 말하라.

[질문]
{question}

[컨텍스트]
{context}

요구사항:
- 간결한 요점 → 필요시 항목화
- 끝에 '출처'로 문서 제목/식별자(있으면) 목록화
"""
)

def _render_context(docs: List[Document], max_docs: int, max_chars: int) -> str:
    parts = []
    for i, d in enumerate(docs[:max_docs], 1):
        meta = d.metadata or {}
        src = meta.get("doc_title") or meta.get("source") or meta.get("doc_id") or "unknown"
        text = (d.page_content or "")[:max_chars]
        parts.append(f"[{i}] {src}\n{text}")
    return "\n\n".join(parts)

def generate_answer(inputs: Dict) -> Dict:
    q = inputs["question"]
    ctx = _render_context(inputs["docs_ranked"], CTX_MAX_DOCS, CTX_MAX_CHARS)
    answer = (answer_prompt | llm_careful | StrOutputParser()).invoke({"question": q, "context": ctx})
    return {"answer": answer, **inputs}

# =========================
# 6) 파이프라인 (LCEL)
# =========================
advanced_rag_chain: Runnable = (
    RunnablePassthrough.assign()
    | RunnableLambda(rewrite_query)
    | RunnableLambda(retrieve_docs)
    | RunnableLambda(apply_rerank)
    | RunnableLambda(generate_answer)
)

def ask_advanced(question: str, history: str = "") -> Dict:
    return advanced_rag_chain.invoke({"question": question, "history": history})

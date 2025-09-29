import os
from typing import Dict, List
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from retriever_chroma import build_chroma_retriever

load_dotenv()

LLM_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "copyright_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# 단일 retriever (가벼운 기본값)
retriever = build_chroma_retriever(
    persist_directory=CHROMA_DIR,
    collection_name=CHROMA_COLLECTION,
    search_type="similarity",
    k=5,
    embedding_model=EMBED_MODEL,
)

def retrieve_docs(inputs: Dict) -> Dict:
    q = inputs["question"]
    docs = retriever.invoke(q)
    return {"docs": docs, **inputs}

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

def render_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("doc_title") or meta.get("source") or meta.get("doc_id") or "unknown"
        parts.append(f"[{i}] {src}\n{d.page_content}")
    return "\n\n".join(parts)

def generate_answer(inputs: Dict, k_ctx: int = 4) -> Dict:
    q = inputs["question"]
    ctx_docs = inputs["docs"][:k_ctx]  # 상위 k 그대로 사용
    ctx = render_context(ctx_docs)
    out = (answer_prompt | llm | StrOutputParser()).invoke({"question": q, "context": ctx})
    # UI 호환을 위해 docs_compressed 키도 채워줌
    return {"answer": out, "docs_compressed": ctx_docs, **inputs}

naive_rag_chain: Runnable = (
    RunnablePassthrough.assign()
    | RunnableLambda(retrieve_docs)
    | RunnableLambda(generate_answer)
)

def ask_naive(question: str, history: str = "") -> Dict:
    return naive_rag_chain.invoke({"question": question, "history": history})

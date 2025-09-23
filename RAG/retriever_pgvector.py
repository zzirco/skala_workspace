# retriever_pgvector.py
from typing import Literal, Optional
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# connection string은 db.CONN_STR 사용하지 않고 문자열만 받도록 분리
def build_pgvector_retriever(
    connection_string: str,
    collection_name: str = "copyright_chunks",  # chunk 테이블/컬렉션 이름
    search_type: Literal["similarity", "mmr"] = "mmr",
    k: int = 5,
    fetch_k: int = 25,  # mmr일 때 후보 개수
    embedding_model: str = "text-embedding-3-small",
) -> BaseRetriever:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vs = PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    if search_type == "mmr":
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
    else:
        return vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

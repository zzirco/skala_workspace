# retriever_chroma.py
from typing import Literal, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever

def build_chroma_retriever(
    persist_directory: str = "./chroma_db",
    collection_name: str = "copyright_chunks",
    search_type: Literal["similarity", "mmr"] = "mmr",
    k: int = 5,
    fetch_k: int = 25,
    embedding_model: str = "text-embedding-3-small",
) -> BaseRetriever:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
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

# retriever_search.py
from sqlalchemy import text
from langchain_openai import OpenAIEmbeddings
from db import engine

emb = OpenAIEmbeddings(model="text-embedding-ada-002")

def search_chunks(query: str, k: int = 5):
    qv = emb.embed_query(query)
    sql = text("""
        SELECT doc_title, doc_id, chunk_index, content,
               1 - (embedding <=> :q::vector) AS cosine_sim
        FROM copyright_chunks
        ORDER BY embedding <=> :q::vector
        LIMIT :k;
    """)
    with engine.connect() as conn:
        return conn.execute(sql, {"q": qv, "k": k}).fetchall()

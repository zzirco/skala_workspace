# ingest_batched.py
import os
import logging
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from db import engine, init_db
from tqdm import tqdm  # pip install tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

TITLE_COL   = "사건명"     # 실제 컬럼명으로 교체
CONTENT_COL = "판례내용"   # 실제 컬럼명으로 교체
ID_COL      = "사건번호"        # 있으면 교체, 없으면 None로 두세요

# 스플리터 (필요 시 조정)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)

def safe_str(x):
    return str(x) if pd.notna(x) else ""

def ingest_data(file_path: str, batch_size: int = 128):
    init_db()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS copyright_chunks (
              id SERIAL PRIMARY KEY,
              doc_title TEXT,
              doc_id    TEXT,
              chunk_index INT,
              content TEXT,
              embedding vector(1536)
            );
        """))

    df = pd.read_excel(file_path)
    logging.info(f"Loaded Excel: rows={len(df)} | cols={list(df.columns)}")

    # 더 빠른/저렴한 최신 모델 권장
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 1) 모든 로우 → 청크로 먼저 변환
    records = []  # [{"title":..., "doc_id":..., "chunk_index":..., "content":...}, ...]
    for _, row in df.iterrows():
        title = safe_str(row.get(TITLE_COL, "")).strip()
        body  = safe_str(row.get(CONTENT_COL, "")).strip()
        docid = safe_str(row.get(ID_COL, "")) if ID_COL and ID_COL != "<ID_COL>" else None
        base  = "\n\n".join([t for t in [title, body] if t])
        if not base:
            continue
        chunks = splitter.split_text(base)
        for i, ch in enumerate(chunks):
            records.append({"title": title, "doc_id": docid, "chunk_index": i, "content": ch})

    logging.info(f"Total chunks to embed: {len(records)}")

    # 2) 배치 임베딩 + 벌크 insert
    inserted = 0
    with engine.begin() as conn:
        for i in tqdm(range(0, len(records), batch_size), desc="Embedding+Insert", unit="batch"):
            batch = records[i:i+batch_size]
            texts = [r["content"] for r in batch]

            # 한 번에 임베딩 요청 (네트워크 왕복 최소화)
            vecs = embeddings.embed_documents(texts)  # List[List[float]]

            params = []
            for r, v in zip(batch, vecs):
                params.append({
                    "dt": r["title"],
                    "di": r["doc_id"],
                    "ci": r["chunk_index"],
                    "ct": r["content"],
                    "e":  v,
                })

            conn.execute(
                text("""
                    INSERT INTO copyright_chunks
                      (doc_title, doc_id, chunk_index, content, embedding)
                    VALUES
                      (:dt, :di, :ci, :ct, :e)
                """),
                params  # executemany
            )
            inserted += len(batch)

    logging.info(f"✅ Done. Inserted chunks: {inserted}")

if __name__ == "__main__":
    ingest_data("Copyright_case_law.xlsx", batch_size=128)

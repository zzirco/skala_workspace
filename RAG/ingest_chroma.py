import os
import argparse
import logging
import pandas as pd
from typing import List, Dict, Iterable, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

# ====== 환경설정 ======
TITLE_COL   = os.getenv("TITLE_COL", "사건명")
CONTENT_COL = os.getenv("CONTENT_COL", "판례내용")
ID_COL      = os.getenv("ID_COL", "사건번호")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION  = os.getenv("CHROMA_COLLECTION", "copyright_chunks")

ADD_BATCH_BY_COUNT = int(os.getenv("ADD_BATCH", "300"))
MAX_TOKENS_PER_REQ = int(os.getenv("MAX_TOKENS_PER_REQ", "290000"))

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)

def safe_str(x) -> str:
    return (str(x) if pd.notna(x) else "").strip()

def make_records(df: pd.DataFrame) -> List[Dict]:
    records = []
    for _, row in df.iterrows():
        title = safe_str(row.get(TITLE_COL, ""))
        body  = safe_str(row.get(CONTENT_COL, ""))
        if not (title or body):
            continue
        docid = safe_str(row.get(ID_COL, "")) if ID_COL != "<ID_COL>" else None
        base = "\n\n".join([t for t in [title, body] if t])
        chunks = splitter.split_text(base)
        for i, ch in enumerate(chunks):
            records.append({
                "text": ch,
                "metadata": {"doc_title": title, "doc_id": docid, "chunk_index": i}
            })
    return records

# ---------- 배치 유틸 ----------
def token_len(s: str) -> int:
    if _enc is None:
        return max(1, int(len(s) * 0.5))
    return len(_enc.encode(s))

def iter_batches_by_tokens(
    texts: List[str],
    metas: List[Dict],
    max_tokens: int,
    fallback_count: int
) -> Iterable[Tuple[List[str], List[Dict]]]:
    """
    tiktoken이 있으면 토큰 총합이 max_tokens를 넘지 않도록 묶고,
    없으면 개수 기준 fallback으로 묶음.
    """
    if not texts:
        return
    if _enc is None:
        for i in range(0, len(texts), fallback_count):
            yield texts[i:i+fallback_count], metas[i:i+fallback_count]
        return

    cur_txt, cur_meta, cur_tok = [], [], 0
    for t, m in zip(texts, metas):
        tl = token_len(t)
        if tl >= max_tokens:
            if cur_txt:
                yield cur_txt, cur_meta
                cur_txt, cur_meta, cur_tok = [], [], 0
            yield [t], [m]
            continue
        if cur_tok + tl > max_tokens and cur_txt:
            yield cur_txt, cur_meta
            cur_txt, cur_meta, cur_tok = [], [], 0
        cur_txt.append(t); cur_meta.append(m); cur_tok += tl

        if len(cur_txt) >= fallback_count:
            yield cur_txt, cur_meta
            cur_txt, cur_meta, cur_tok = [], [], 0

    if cur_txt:
        yield cur_txt, cur_meta

def ingest_chroma(xlsx_path: str, reset: bool = False):
    logging.info(f"Loading: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    logging.info(f"Rows={len(df)} | Cols={list(df.columns)}")

    recs = make_records(df)
    logging.info(f"Total chunks={len(recs)}")

    if reset and os.path.exists(PERSIST_DIR):
        logging.info(f"Reset persist dir: {PERSIST_DIR}")
        import shutil; shutil.rmtree(PERSIST_DIR)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Chroma VectorStore 생성/오픈
    vs = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    texts = [r["text"] for r in recs]
    metas = [r["metadata"] for r in recs]
    if not texts:
        logging.warning("No chunks generated. Check column names.")
        return

    added = 0
    total = len(texts)
    for t_batch, m_batch in iter_batches_by_tokens(texts, metas, MAX_TOKENS_PER_REQ, ADD_BATCH_BY_COUNT):
        vs.add_texts(texts=t_batch, metadatas=m_batch)
        added += len(t_batch)
        logging.info(f"Chroma add {added}/{total} (this call: {len(t_batch)})")

    vs.persist()
    logging.info(f"✅ Ingested into Chroma: collection={COLLECTION}, dir={PERSIST_DIR}, added={len(texts)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="Copyright_case_law.xlsx")
    ap.add_argument("--reset", action="store_true", help="기존 ChromaDB 디렉토리 초기화")
    args = ap.parse_args()
    ingest_chroma(args.file, reset=args.reset)

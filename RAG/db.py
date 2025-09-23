import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR)

def init_db():
  with engine.connect() as conn:
    conn.execute(text("""
      CREATE TABLE IF NOT EXISTS rag.copyright_cases (
        id SERIAL PRIMARY KEY,
        title TEXT,
        content TEXT,
        embedding vector(1536)
      );
    """))
    conn.commit()

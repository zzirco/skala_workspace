import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수에서 DB 접속 정보 읽기
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# 1단계: 리뷰 데이터 임베딩
reviews = [
    "배송이 빠르고 제품도 좋아요.",
    "품질이 기대 이상입니다!",
    "생각보다 배송이 오래 걸렸어요.",
    "배송은 느렸지만 포장은 안전했어요.",
    "아주 만족스러운 제품입니다."
]

# SentenceTransformer 모델 로드
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# 각 리뷰를 벡터화
embeddings = model.encode(reviews)

# 2단계: PostgreSQL 연결
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# review_vectors 테이블 생성
cur.execute("""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS review_vectors (
    id SERIAL PRIMARY KEY,
    review TEXT,
    embedding VECTOR(384)
);
""")
conn.commit()

# 3단계: 리뷰 및 벡터 저장
cur.execute("TRUNCATE review_vectors;")  # 중복 저장 방지용 초기화
for review, emb in zip(reviews, embeddings):
    emb_list = emb.tolist()  # numpy → list 변환
    cur.execute(
        "INSERT INTO review_vectors (review, embedding) VALUES (%s, %s)",
        (review, emb_list)
    )
conn.commit()

print("리뷰 및 임베딩 저장 완료")

# 4단계: 유사도 검색
query = "배송이 느렸어요"
query_vec = model.encode([query])[0].tolist()

cur.execute(
    """
    SELECT review, 1 - (embedding <=> %s::vector) AS cosine_similarity
    FROM review_vectors
    ORDER BY cosine_similarity DESC
    LIMIT 3;
    """,
    (query_vec,)
)

results = cur.fetchall()
print("유사 리뷰 검색 결과:")
for review, score in results:
    print(f"- {review} (유사도: {score:.4f})")

cur.close()
conn.close()

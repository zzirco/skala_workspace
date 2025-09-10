
-- Step 1: 테이블 생성
DROP TABLE IF EXISTS post_reviews;

CREATE TABLE post_reviews (
  review_id SERIAL PRIMARY KEY,
  post_id TEXT,
  review_text TEXT,
  metadata JSONB
);

-- Step 2: 데이터 삽입
INSERT INTO post_reviews (post_id, review_text, metadata) VALUES
('POST001', '이 포스트는 인사이트가 풍부했어요!', '{"topic": "AI", "sentiment": "positive", "language": "ko"}'),
('POST002', '내용이 다소 어렵고 추상적입니다.', '{"topic": "philosophy", "sentiment": "negative", "language": "en"}'),
('POST003', '짧고 명확해서 유익했어요!', '{"topic": "productivity", "sentiment": "positive", "language": "ko"}'),
('POST004', '그저 그런 느낌이었어요.', '{"topic": "lifestyle", "sentiment": "neutral", "language": "en"}');

-- Step 3: GIN 인덱스 생성
CREATE INDEX idx_post_reviews_metadata ON post_reviews USING GIN (metadata);



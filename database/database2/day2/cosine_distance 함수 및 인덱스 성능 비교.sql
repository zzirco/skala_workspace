-- 1단계 : 유사도 쿼리 기본 구조(코사인 거리)
SELECT title, content, embedding_vector
FROM design_doc
ORDER BY embedding_vector <=> (SELECT embedding_vector FROM design_doc WHERE title like '생성 제목 00001')
LIMIT 5;

-- 2단계 : 성능 비교 (인덱스 없이 실행계획)
EXPLAIN ANALYZE
SELECT title, content, embedding_vector
FROM design_doc
ORDER BY embedding_vector <=> (SELECT embedding_vector FROM design_doc WHERE title like '생성 제목 00001')
LIMIT 5;

-- 3단계 : ivfflat 인덱스 생성 (코사인 거리 전용)
CREATE INDEX ON design_doc
USING ivfflat (embedding_vector vector_cosine_ops)
WITH (lists = 100);

-- 4단계 : 인덱스 적용 후 실행계획 확인
EXPLAIN ANALYZE
SELECT title, content, embedding_vector
FROM design_doc
ORDER BY embedding_vector <=> (SELECT embedding_vector FROM design_doc WHERE title like '생성 제목 00001')
LIMIT 5;

-- 5단계 : LIMIT 값 증가 후 검색 시간 비교
EXPLAIN ANALYZE
SELECT title, content, embedding_vector
FROM design_doc
ORDER BY embedding_vector <=> (SELECT embedding_vector FROM design_doc WHERE title like '생성 제목 00001')
LIMIT 10000;

-- 6단계: vector_l2_ops vs vector_cosine_ops 차이 실험
-- L2 거리 인덱스 생성
CREATE INDEX ON design_doc_l2
USING ivfflat (embedding_vector vector_l2_ops)
WITH (lists = 100);

-- 각각 실행계획 비교
-- l2
EXPLAIN ANALYZE
SELECT title, content, embedding_vector
FROM design_doc
ORDER BY embedding_vector <-> (SELECT embedding_vector FROM design_doc WHERE title like '생성 제목 00001')
LIMIT 10;

-- cosine
EXPLAIN ANALYZE
SELECT title, content, embedding_vector
FROM design_doc
ORDER BY embedding_vector <=> (SELECT embedding_vector FROM design_doc WHERE title like '생성 제목 00001')
LIMIT 10;


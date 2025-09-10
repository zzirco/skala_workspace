-- =========================================
-- AI 서비스 리뷰 벤치마크용 대량 더미 데이터 시드
-- =========================================

-- psql에서 크기 조정(원하면 수정)
-- \set creators 10000
-- \set reviews  1000000

-- psql 변수 기본값 설정 (psql에서 미지정 시)
DO $$
BEGIN
  IF current_setting('server_version_num')::int >= 90600 THEN
    -- no-op: just to ensure block runs
    NULL;
  END IF;
END$$;

-- 변수 유사 기능: psql에서 지정 안 했다면 기본값 사용
-- (아래에서는 COALESCE로 처리)
WITH defaults AS (
  SELECT
    COALESCE(NULLIF(current_setting('ai.creators', true), ''), '10000')::int  AS creators,
    COALESCE(NULLIF(current_setting('ai.reviews',  true), ''), '1000000')::bigint AS reviews
)
SELECT set_config('ai.creators', creators::text, false),
       set_config('ai.reviews',  reviews::text,  false)
FROM defaults;

-- 성능을 위한 일시 설정(세션 한정)
SET work_mem = '256MB';
SET maintenance_work_mem = '512MB';
SET random_page_cost = 1.1;
SET effective_cache_size = '4GB';

BEGIN;

-- 기존 테이블 제거 및 재생성 (질문 제공 DDL 준수)
DROP TABLE IF EXISTS ai_service_reviews;
DROP TABLE IF EXISTS ai_service_creators;

CREATE TABLE ai_service_creators (
    creator_id SERIAL PRIMARY KEY,
    creator_name TEXT
);

CREATE TABLE ai_service_reviews (
    review_id  SERIAL PRIMARY KEY,
    creator_id INTEGER REFERENCES ai_service_creators(creator_id),
    rating     INTEGER,  -- 1~5점
    review_text TEXT
);

-- -----------------------------
-- 1) 기획자 1..N 생성
-- -----------------------------
-- creators = current_setting('ai.creators')::int
INSERT INTO ai_service_creators (creator_name)
SELECT 'Creator ' || gs::text
FROM generate_series(1, current_setting('ai.creators')::int) AS gs;

-- -----------------------------
-- 2) 리뷰 1..M 생성
-- -----------------------------
-- 분포 설명:
--  - creator_id: random()^2 로 편향(상위 소수의 인기 기획자에게 더 많은 리뷰가 몰리는 효과)
--  - rating: 현실적 비율(1:5%, 2:10%, 3:30%, 4:35%, 5:20%)
--  - review_text: 간단한 문구 + 일련번호
-- 리뷰 대량 생성 (수정본)
WITH params AS (
  SELECT
    current_setting('ai.creators')::int    AS creators,
    current_setting('ai.reviews')::bigint  AS reviews
)
INSERT INTO ai_service_reviews (creator_id, rating, review_text)
SELECT
  -- creator_id: 1 ~ creators, 인기 편향 적용 (random()^2로 낮은 id에 집중)
  (1 + floor(power(random(), 2) * p.creators))::int AS creator_id,

  -- rating: 1 ~ 5 균등 분포
  (floor(random() * 5) + 1)::int AS rating,

  'auto review #' || gs.i::text AS review_text
FROM params p
CROSS JOIN LATERAL generate_series(1, p.reviews) AS gs(i);


-- -----------------------------
-- 3) 인덱스 & 통계 수집
-- -----------------------------
CREATE INDEX idx_ai_reviews_creator ON ai_service_reviews (creator_id);
CREATE INDEX idx_ai_reviews_rating  ON ai_service_reviews (rating);

ANALYZE ai_service_creators;
ANALYZE ai_service_reviews;

COMMIT;

-- -----------------------------
-- 4) 검증(행 수와 샘플)
-- -----------------------------
-- 총 개수 확인
SELECT
  (SELECT COUNT(*) FROM ai_service_creators) AS creators_cnt,
  (SELECT COUNT(*) FROM ai_service_reviews)  AS reviews_cnt;

-- 상위 5명(리뷰 수 많은 순)
SELECT c.creator_id, c.creator_name, COUNT(*) AS review_cnt
FROM ai_service_reviews r
JOIN ai_service_creators c USING (creator_id)
GROUP BY c.creator_id, c.creator_name
ORDER BY review_cnt DESC
LIMIT 5;

-- 평점 분포 확인
SELECT rating, COUNT(*) AS cnt, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM ai_service_reviews
GROUP BY rating
ORDER BY rating;

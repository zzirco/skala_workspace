WITH base AS (
  SELECT
    r.creator_id,
    AVG(r.rating) OVER (PARTITION BY r.creator_id)::numeric(3,2) AS avg_rating,
    COUNT(*)      OVER (PARTITION BY r.creator_id)               AS review_cnt
  FROM ai_service_reviews r
),
dedup AS (
  SELECT DISTINCT ON (creator_id)
    creator_id, avg_rating, review_cnt
  FROM base
  WHERE review_cnt >= 2
)
SELECT
  c.creator_id,
  c.creator_name,
  d.avg_rating,
  d.review_cnt,
  ROW_NUMBER() OVER (
    ORDER BY d.avg_rating DESC, d.review_cnt DESC, c.creator_id
  ) AS rank
FROM dedup d
JOIN ai_service_creators c USING (creator_id)
ORDER BY rank;
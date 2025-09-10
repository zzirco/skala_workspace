WITH stats AS (
  SELECT
    r.creator_id,
    AVG(r.rating)::numeric(3,2) AS avg_rating,
    COUNT(*)                    AS review_cnt
  FROM ai_service_reviews r
  GROUP BY r.creator_id
  HAVING COUNT(*) >= 2
)
SELECT
  c.creator_id,
  c.creator_name,
  s.avg_rating,
  s.review_cnt,
  ROW_NUMBER() OVER (
    ORDER BY s.avg_rating DESC, s.review_cnt DESC, c.creator_id
  ) AS rank
FROM stats s
JOIN ai_service_creators c USING (creator_id)
ORDER BY rank;
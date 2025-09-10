select * from post_reviews pr
where pr.metadata ->> 'sentiment' = 'negative';
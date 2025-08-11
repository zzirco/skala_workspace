select * from post_reviews pr
where not (pr.metadata @> '{"topic": "productivity"}');
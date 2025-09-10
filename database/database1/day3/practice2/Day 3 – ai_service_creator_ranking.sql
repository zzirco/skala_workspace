
-- ✅ 실습 2: WITH (CTE) + 집계로 인기 기획자 추출

-- 기존 테이블 제거
DROP TABLE IF EXISTS ai_service_reviews;
DROP TABLE IF EXISTS ai_service_creators;

-- 기획자 테이블 생성
CREATE TABLE ai_service_creators (
    creator_id SERIAL PRIMARY KEY,
    creator_name TEXT
);

-- 리뷰 테이블 생성
CREATE TABLE ai_service_reviews (
    review_id SERIAL PRIMARY KEY,
    creator_id INTEGER REFERENCES ai_service_creators(creator_id),
    rating INTEGER,  -- 1~5점
    review_text TEXT
);

-- 기획자 데이터 삽입
INSERT INTO ai_service_creators (creator_name) VALUES
('Alice Kim'),
('Brian Lee'),
('Clara Park'),
('David Choi');

-- 리뷰 데이터 삽입
INSERT INTO ai_service_reviews (creator_id, rating, review_text) VALUES
(1, 5, '서비스가 직관적이고 좋았습니다.'),
(1, 4, '빠르게 응답했어요.'),
(2, 3, '기능이 부족해요.'),
(2, 2, '사용성이 떨어져요.'),
(2, 4, '업데이트 기대합니다.'),
(3, 5, '딥러닝 기능이 인상 깊었어요.'),
(3, 5, '추천 정확도가 높아요.'),
(4, 3, '보통이에요.'),
(4, 2, '불편했어요.');



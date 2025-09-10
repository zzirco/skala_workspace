
-- ✅ 실습 1: 키워드 기반 서비스 기획안 검색 쿼리

-- 테이블 삭제 및 생성
DROP TABLE IF EXISTS ai_service_plans;

CREATE TABLE ai_service_plans (
    id SERIAL PRIMARY KEY,
    service_name TEXT,
    category TEXT,           -- 예: '헬스케어', '금융', '교육', '리테일'
    description_text TEXT    -- 서비스 개요 설명
);

-- 예시 데이터 삽입
INSERT INTO ai_service_plans (service_name, category, description_text) VALUES
('SmartFit', '헬스케어', 'AI를 활용한 개인 맞춤형 운동 코칭 서비스'),
('EduBot', '교육', '학생의 학습 패턴을 분석하여 추천 커리큘럼 제공'),
('FinGuard', '금융', '소비 내역 분석을 통한 금융 사기 탐지 서비스'),
('RetailMate', '리테일', '매장 내 고객 행동을 분석하여 상품 배치 최적화'),
('HealthGuard', '헬스케어', '건강 기록 기반의 조기 질병 예측 AI 시스템');


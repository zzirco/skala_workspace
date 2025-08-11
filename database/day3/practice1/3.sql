select service_name as 서비스, category as 카테고리, description_text as 설명
from ai_service_plans
where description_text ilike '%추천%'
and category = '교육';
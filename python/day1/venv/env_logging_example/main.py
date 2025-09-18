import os
import logging
from dotenv import load_dotenv

# ================================
# 1. .env 파일 로드
# ================================
# .env 파일을 로드하여 환경 변수를 현재 실행 환경에 주입
load_dotenv()

# .env 파일에서 변수 가져오기
log_level = os.getenv("LOG_LEVEL")
app_name = os.getenv("APP_NAME")

# ================================
# 2. 로그 설정
# ================================
# 로그 포맷 정의: 시간 | 로그레벨 | 메시지
log_format = "%(asctime)s | %(levelname)s | %(message)s"

# 로그 레벨을 문자열로 읽어와 logging 모듈의 상수로 변환
numeric_level = getattr(logging, log_level.upper(), logging.INFO)

# 루트 로거 기본 설정
logging.basicConfig(
    level=numeric_level,           # 로그 레벨
    format=log_format,             # 로그 출력 포맷
    handlers=[
        logging.StreamHandler(),   # 콘솔 출력
        logging.FileHandler("app.log", encoding="utf-8")  # 파일 출력
    ]
)

# ================================
# 3. 로그 출력 테스트
# ================================

# INFO 로그 출력
logging.info("앱 실행 시작")

# DEBUG 로그 출력
logging.debug("환경 변수 로딩 완료")

# ERROR 로그 출력 (예외 처리 예시)
try:
    1 / 0   # ZeroDivisionError 발생
except Exception as e:
    logging.error("에러: %s", e)

# ================================
# 4. 프로그램 실행 확인용 출력
# ================================
print(f"앱 이름: {app_name}")
print(f"현재 로그 레벨: {log_level}")

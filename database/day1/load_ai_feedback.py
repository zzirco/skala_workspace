import psycopg2
import csv
from pathlib import Path

# ====== DB 연결 설정 ======
DB_CONFIG = {
    "host": "localhost",   # 필요시 '127.0.0.1'로 변경
    "port": 5432,
    "dbname": "ai_feedback",
    "user": "postgres",
    "password": "a"
}

CSV_FILE = Path(r"C:\workspace\database\ai_feedback.csv")  # CSV 경로 수정
SCHEMA = "ai_feedback"  # 스키마 명 (대소문자/따옴표 주의)

def q(table_name: str) -> str:
    """스키마가 포함된 정규 테이블명 반환"""
    return f"{SCHEMA}.{table_name}"

# ====== 헬퍼 함수 ======
def clean_tags(tags_str):
    """CSV tags 문자열을 TEXT[]로 변환"""
    if not tags_str or tags_str.strip() == "":
        return []
    return [tag.strip() for tag in tags_str.split(",") if tag.strip()]

# ====== 메인 ======
def main():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    with open(CSV_FILE, "r", encoding="cp949") as f:  # 인코딩 환경에 맞게 조정
        reader = csv.DictReader(f)
        for row in reader:
            feedback_id   = row["feedback_id"].strip()
            model_name    = row["model_name"].strip()
            user_name     = row["user_name"].strip()
            prompt_text   = row["prompt_text"].strip()
            response_text = row["response_text"].strip()
            rating        = float(row["rating"]) if row["rating"] else 0.0
            tags_array    = clean_tags(row["tags"])

            # 1) app_user 삽입
            cur.execute(f"""
                INSERT INTO {q('app_user')} (user_name)
                VALUES (%s)
                ON CONFLICT (user_name) DO NOTHING
                RETURNING user_id
            """, (user_name,))
            if cur.rowcount > 0:
                user_id = cur.fetchone()[0]
            else:
                cur.execute(f"SELECT user_id FROM {q('app_user')} WHERE user_name = %s", (user_name,))
                user_id = cur.fetchone()[0]

            # 2) model 삽입
            cur.execute(f"""
                INSERT INTO {q('model')} (model_name)
                VALUES (%s)
                ON CONFLICT (model_name) DO NOTHING
                RETURNING model_id
            """, (model_name,))
            if cur.rowcount > 0:
                model_id = cur.fetchone()[0]
            else:
                cur.execute(f"SELECT model_id FROM {q('model')} WHERE model_name = %s", (model_name,))
                model_id = cur.fetchone()[0]

            # 3) feedback 삽입
            cur.execute(f"""
                INSERT INTO {q('feedback')}
                    (feedback_id, user_id, model_id, prompt_text, response_text, rating, tags_array)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (feedback_id) DO NOTHING
                RETURNING feedback_pk
            """, (feedback_id, user_id, model_id, prompt_text, response_text, rating, tags_array))
            if cur.rowcount > 0:
                feedback_pk = cur.fetchone()[0]
            else:
                cur.execute(f"SELECT feedback_pk FROM {q('feedback')} WHERE feedback_id = %s", (feedback_id,))
                feedback_pk = cur.fetchone()[0]

            # 4) tag & feedback_tag 삽입
            for tag_name in tags_array:
                cur.execute(f"""
                    INSERT INTO {q('tag')} (tag_name)
                    VALUES (%s)
                    ON CONFLICT (tag_name) DO NOTHING
                    RETURNING tag_id
                """, (tag_name,))
                if cur.rowcount > 0:
                    tag_id = cur.fetchone()[0]
                else:
                    cur.execute(f"SELECT tag_id FROM {q('tag')} WHERE tag_name = %s", (tag_name,))
                    tag_id = cur.fetchone()[0]

                cur.execute(f"""
                    INSERT INTO {q('feedback_tag')} (feedback_pk, tag_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (feedback_pk, tag_id))

    conn.commit()
    cur.close()
    conn.close()
    print("CSV 데이터 적재 완료!")

if __name__ == "__main__":
    main()

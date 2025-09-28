# streamlit_app.py
import os
import json
import time
from pathlib import Path
import streamlit as st
from typing import Dict, Any
from rag_client import ask, RagResult

st.set_page_config(
    page_title="저작권 법률자문 챗봇 (RAG)",
    page_icon="⚖️",
    layout="wide",
)

# -------------------- 로그 유틸 --------------------
EVAL_LOG_PATH = os.getenv("EVAL_LOG_PATH", "eval_logs.jsonl")
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "eval_dataset.jsonl")

def append_eval_log(item: dict):
    """대화 1턴(question/answer/contexts 등)을 JSONL 한 줄로 추가 저장."""
    Path(EVAL_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def build_eval_dataset_from_logs(log_path: str, out_path: str, max_items: int = 500) -> int:
    """eval_logs.jsonl → eval_dataset.jsonl (최소 포맷: question만)"""
    seen = set()
    out = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(out) >= max_items:
                    break
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                q = (item.get("question") or "").strip()
                a = (item.get("answer") or "").strip()
                ctx = item.get("contexts") or []
                if not q or not a or not ctx:
                    continue
                if q in seen:  # 질문 중복 제거
                    continue
                seen.add(q)
                out.append({
                    "question": q,
                    "reference_answer": "",
                    "gold_contexts": []
                })
        if out:
            Path(out_path).write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in out),
                encoding="utf-8"
            )
        return len(out)
    except FileNotFoundError:
        return 0

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("⚙️ 설정")
    backend = os.getenv("RAG_BACKEND", "python").lower()
    st.write(f"**Backend**: `{backend}`")
    if backend == "api":
        st.write(f"**FASTAPI_URL**: `{os.getenv('FASTAPI_URL','http://127.0.0.1:8000')}`")

    # 파이프라인 선택 (advanced / naive)
    pipeline = st.radio("파이프라인 선택", options=["advanced", "naive"], index=0, horizontal=True)

    show_route = st.checkbox("라우팅/전략 표시", value=True)
    show_sources = st.checkbox("근거/출처 보기", value=True)

    st.markdown("---")
    st.markdown("### 🧪 평가 데이터셋")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📥 로그에서 평가셋 만들기"):
            n = build_eval_dataset_from_logs(EVAL_LOG_PATH, EVAL_DATASET_PATH, max_items=1000)
            if n > 0:
                st.success(f"생성 완료: {EVAL_DATASET_PATH} (샘플 {n}개)")
            else:
                st.warning("생성할 로그가 없거나 유효한 항목이 없습니다.")
    with col_b:
        if st.button("🗑️ 평가 로그 초기화"):
            try:
                Path(EVAL_LOG_PATH).unlink(missing_ok=True)
                st.success("eval_logs.jsonl 초기화 완료")
            except Exception as e:
                st.error(f"초기화 실패: {e}")

    st.markdown("---")
    if st.button("🧹 대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# -------------------- Session State --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"|"assistant","content":"..."}]

# -------------------- Header --------------------
st.markdown("## ⚖️ 저작권 법률자문 챗봇")
st.caption("ChromaDB · RAG Pipelines (Advanced / Naive) + 대화 로그 기반 평가셋 생성")

# -------------------- Render History --------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------- User Input --------------------
user_q = st.chat_input("질문을 입력하세요. 예) 저작권 침해 손해배상 산정 기준은?")
if user_q:
    # 사용자 메시지 표시/저장
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # 백엔드 호출
    with st.chat_message("assistant"):
        with st.spinner(f"{pipeline} 파이프라인 실행 중..."):
            try:
                result: RagResult = ask(user_q, st.session_state.messages, pipeline=pipeline)
                answer_text = result.answer or "_응답이 비어 있습니다._"

                # 답변 본문
                st.markdown(answer_text)

                # 라우팅/전략 (Advanced에서만 있을 수 있음)
                if show_route and result.route:
                    st.caption(f"**Route/Strategy**: `{result.route}`")

                # 근거/출처
                if show_sources:
                    if result.docs:
                        st.markdown("#### 🔎 사용된 컨텍스트")
                        for i, d in enumerate(result.docs, 1):
                            meta = d.get("metadata") or {}
                            title = meta.get("doc_title") or meta.get("source") or meta.get("doc_id") or "unknown"
                            with st.expander(f"[{i}] {title}"):
                                st.write("**metadata**:", meta)
                                st.write(d.get("preview") or "")
                    else:
                        st.caption("컨텍스트가 비어 있습니다.")

                # 어시스턴트 메시지 저장
                st.session_state.messages.append({"role": "assistant", "content": answer_text})

                # ✅ 대화 로그 자동 저장 (RAGAS 준비용)
                try:
                    append_eval_log({
                        "ts": int(time.time()),
                        "pipeline": pipeline,                 # "advanced" | "naive"
                        "question": user_q,
                        "answer": answer_text,
                        "contexts": [d.get("preview","") for d in result.docs],
                        "route": result.route or "",
                        "history": "\n".join(
                            [f"{m['role']}:{m['content']}" for m in st.session_state.messages if m["role"]=="user"]
                        )[:2000]
                    })
                except Exception:
                    # 로그 실패는 조용히 무시
                    pass

            except Exception as e:
                msg = str(e)
                if "insufficient_quota" in msg or "Error code: 429" in msg:
                    st.error("OpenAI 쿼터가 소진되었습니다. .env에서 로컬 백엔드(HF/Ollama)로 전환하거나, 결제/크레딧을 확인하세요.")
                else:
                    st.error(f"오류가 발생했습니다: {e}")

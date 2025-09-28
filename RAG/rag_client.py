# rag_client.py
"""
RAG 백엔드 호출 추상화 모듈.

동작 모드(환경변수):
- RAG_BACKEND = "python" | "api"        (기본: python)
- FASTAPI_URL = "http://127.0.0.1:8000" (api 모드일 때)
- USE_CHAT_API = "true" | "false"       (api 모드에서 /chat 사용 여부)

파이프라인 선택:
- ask(..., pipeline="advanced" | "naive")
  - python 모드: rag_advanced.ask_advanced / rag_naive.ask_naive 직접 호출
  - api    모드: 서버가 pipeline을 해석하거나, /ask_advanced /ask_naive 엔드포인트가 있어야 함
"""

import os
from typing import Dict, Any, List, Optional

BACKEND = os.getenv("RAG_BACKEND", "python").lower()
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
USE_CHAT_API = os.getenv("USE_CHAT_API", "true").lower() == "true"


class RagResult:
    def __init__(self, answer: str, route: Optional[str], docs: List[Dict[str, Any]]):
        self.answer = answer or ""
        self.route = route
        # docs: [{"metadata": {...}, "preview": "..."}]
        self.docs = docs or []


def _history_to_text(history_msgs: List[Dict[str, str]]) -> str:
    """Streamlit 세션 메시지 -> 서버로 전달할 history 문자열."""
    lines: List[str] = []
    for m in history_msgs:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자:" if role == "user" else "어시스턴트:"
        lines.append(f"{prefix} {content}")
    return "\n".join(lines[-20:])  # 최근 20턴만 유지


# -----------------------------
# Python 모드 (동일 프로세스)
# -----------------------------
def _ask_python(q: str, history_msgs: List[Dict[str, str]], pipeline: str) -> RagResult:
    history_text = _history_to_text(history_msgs)
    if pipeline == "naive":
        from rag_naive import ask_naive  # 지연 임포트
        out: Dict[str, Any] = ask_naive(q, history_text)
    else:
        from rag_advanced import ask_advanced
        out = ask_advanced(q, history_text)

    # UI 호환: 여러 키 중 우선순위 폴백
    src_docs = out.get("docs_compressed") or out.get("docs_ranked") or out.get("docs") or []
    used = [
        {
            "metadata": (getattr(d, "metadata", None) or {}),
            "preview": (getattr(d, "page_content", None) or "")[:500],
        }
        for d in src_docs
    ]
    return RagResult(answer=out.get("answer", ""), route=out.get("route"), docs=used)


# -----------------------------
# API 모드 (FastAPI 호출)
# -----------------------------
def _ask_api(q: str, history_msgs: List[Dict[str, str]], pipeline: str) -> RagResult:
    import requests
    headers = {"Content-Type": "application/json"}

    if USE_CHAT_API:
        # 서버가 /chat에서 pipeline 파라미터를 받아 처리하도록 구현되어 있어야 합니다.
        url = f"{FASTAPI_URL}/chat"
        payload = {"q": q, "pipeline": pipeline}
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return RagResult(
            answer=data.get("answer", ""),
            route=data.get("route"),
            docs=data.get("used_docs", []),
        )
    else:
        # 서버에 /ask_advanced, /ask_naive 엔드포인트가 있는 경우에만 사용 가능
        url = f"{FASTAPI_URL}/ask_advanced" if pipeline != "naive" else f"{FASTAPI_URL}/ask_naive"
        history_text = _history_to_text(history_msgs)
        resp = requests.get(url, params={"q": q, "history": history_text}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return RagResult(
            answer=data.get("answer", ""),
            route=data.get("route"),
            docs=data.get("used_docs", []),
        )


def ask(q: str, history_msgs: List[Dict[str, str]], pipeline: str = "advanced") -> RagResult:
    if BACKEND == "api":
        return _ask_api(q, history_msgs, pipeline)
    return _ask_python(q, history_msgs, pipeline)

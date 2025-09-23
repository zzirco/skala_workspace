# main.py
from typing import Any, Dict, List, Optional, Deque, Tuple
from collections import deque
from fastapi import FastAPI, Body, Header, Response
from pydantic import BaseModel, Field
from uuid import uuid4
from db import init_db
from rag_advanced import ask_advanced
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# 앱 메타: (기존 그대로)
# -------------------------------
app = FastAPI(
    title="Copyright RAG API",
    version="1.1.0",
    description=(
        "로컬 PostgreSQL + pgvector 저작권 판례 기반 Advanced RAG.\n"
        "Pre: Rewrite/Routing/Expansion/Transformation | Post: Reranker/Compression"
    ),
    openapi_tags=[
        {"name": "RAG", "description": "Advanced RAG 엔드포인트"},
        {"name": "Chat", "description": "세션 기반 대화형 엔드포인트"},
        {"name": "Health", "description": "상태 점검"},
    ],
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

# -------------------------------
# (참고) 기존 ask_advanced GET/POST는 그대로 유지 가능
# -------------------------------
class DocPreview(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    preview: str

class AskAdvancedResponse(BaseModel):
    question: str
    answer: str
    route: Optional[str] = None
    used_docs: List[DocPreview] = Field(default_factory=list)

# -------------------------------
# 세션 관리 (데모: 인메모리)
# 실제 운영: Redis/DB 권장
# -------------------------------
# history 저장 형태: deque[("user", text) | ("assistant", text)]
SessionHistory = Deque[Tuple[str, str]]
SESSION_STORE: Dict[str, SessionHistory] = {}

MAX_TURNS = 12          # 최근 N턴만 유지
MAX_CHARS_HISTORY = 4000  # 과한 비용 방지 (토큰 기준이면 더 좋음)

def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in SESSION_STORE:
        return session_id
    sid = session_id or str(uuid4())
    SESSION_STORE.setdefault(sid, deque(maxlen=MAX_TURNS * 2))  # user/assistant 페어로 ×2
    return sid

def history_to_text(sid: str) -> str:
    """간단 문자열 합치기(운영: 요약(summarize)로 대체 가능)."""
    hist = SESSION_STORE.get(sid, deque())
    text_parts = []
    for role, msg in hist:
        prefix = "사용자:" if role == "user" else "어시스턴트:"
        text_parts.append(f"{prefix} {msg}")
    joined = "\n".join(text_parts)
    # 과도하면 뒷부분 우선 남겨둠
    if len(joined) > MAX_CHARS_HISTORY:
        return joined[-MAX_CHARS_HISTORY:]
    return joined

def append_turn(sid: str, user_q: str, assistant_a: str):
    SESSION_STORE[sid].append(("user", user_q))
    SESSION_STORE[sid].append(("assistant", assistant_a))

# -------------------------------
# Chat API: q만 받으면 됨
# -------------------------------
class ChatRequest(BaseModel):
    q: str = Field(..., description="사용자 질문(한 줄)", examples=["저작권 침해 손해배상 산정 기준은?"])

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    route: Optional[str] = None
    used_docs: List[DocPreview] = Field(default_factory=list)

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="세션 기반 대화형 RAG (사용자는 q만 전송)",
    description=(
        "클라이언트는 q만 보내면 됩니다. 서버가 세션별로 대화 이력을 관리하고 "
        "Query Rewrite에 자동으로 history를 주입합니다.\n\n"
        "세션 식별은 `X-Session-Id` 요청 헤더로 합니다. 없으면 서버가 생성해 응답 헤더에 반환합니다."
    ),
)
def chat_endpoint(
    payload: ChatRequest = Body(...),
    response: Response = None,
    x_session_id: Optional[str] = Header(default=None, convert_underscores=False, description="세션 ID (선택)"),
):
    # 1) 세션 확보
    sid = get_or_create_session(x_session_id)
    # 응답 헤더에 세션ID 노출 (최초 생성 시 클라이언트가 이후 요청에 헤더로 재사용)
    response.headers["X-Session-Id"] = sid

    # 2) 서버가 history 구성
    history_text = history_to_text(sid)

    # 3) Advanced RAG 실행 (history는 서버가 주입)
    out = ask_advanced(payload.q, history_text)

    # 4) 이력 업데이트
    ans = out.get("answer", "")
    append_turn(sid, payload.q, ans)

    # 5) 최소 응답 객체 구성
    used = [
        DocPreview(metadata=(d.metadata or {}), preview=(d.page_content or "")[:200])
        for d in out.get("docs_compressed", [])
    ]
    return ChatResponse(
        session_id=sid,
        answer=ans,
        route=out.get("route"),
        used_docs=used,
    )

# -------------------------------
# Health
# -------------------------------
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

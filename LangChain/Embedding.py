from dotenv import load_dotenv
import os
import re
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts import FewShotPromptTemplate, PromptTemplate as LegacyPromptTemplate

# ============================== Bootstrap ==============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HAS_KEY = bool(OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if HAS_KEY else None
if not HAS_KEY:
    print("[INFO] OPENAI_API_KEY가 없어 LLM 호출은 생략됩니다.")
emb = OpenAIEmbeddings() if HAS_KEY else None

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def show_triplet(user_input: str, prompt_preview: str, response_text: str | None):
    print("\n[USER INPUT]")
    print(user_input)
    print("\n[PROMPT]")
    print(prompt_preview)
    if response_text is not None:
        print("\n[LLM RESPONSE]")
        print(response_text)

# ========================= Loaders & Splitter ==========================
DOC_PATHS = [
    "./finance-keywords.txt",
    "./nlp-keywords.txt",
    "./SPRi AI Brief_9월호_산업동향_0909_F.pdf",
    "./IS-208 과학을 위한 AI(AI4Science) 연구의 패러다임을 바꾸다.pdf",
]

def load_documents(paths: List[str]):
    docs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext == ".txt":
                docs += TextLoader(p, encoding="utf-8").load()
            elif ext == ".pdf":
                docs += PyPDFLoader(p).load()
            else:
                print(f"[WARN] Unsupported file type skipped: {p}")
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n", " ", ""],  # 헤더/문단 우선
    )
    return splitter.split_documents(docs)

all_docs = load_documents(DOC_PATHS)
chunks = split_documents(all_docs)

# ============================ FAISS Vector DB ===========================
INDEX_DIR = Path("./faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_NAME = "ai_docs_faiss"

vectorstore = None
if emb:
    # 인덱스가 있으면 로드, 없으면 빌드
    index_path = INDEX_DIR / INDEX_NAME
    if (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists():
        vectorstore = FAISS.load_local(str(index_path), emb, allow_dangerous_deserialization=True)
        print(f"[INFO] Loaded existing FAISS index from {index_path}")
    else:
        print("[INFO] Building FAISS index ...")
        vectorstore = FAISS.from_documents(chunks, emb)
        vectorstore.save_local(str(index_path))
        print(f"[INFO] Saved FAISS index to {index_path}")
else:
    print("[WARN] No embeddings available; FAISS retrieval disabled (will fallback to keyword scoring).")

# =================== Retrieval (FAISS with graceful fallback) ===================
def pick_context_by_faiss(query: str, k: int = 4) -> str:
    """임베딩 검색으로 컨텍스트 구성. 실패 시 빈 문자열 반환."""
    if not vectorstore:
        return ""
    docs = vectorstore.similarity_search(query, k=k)
    selected = []
    for d in docs:
        meta = d.metadata or {}
        src = os.path.basename(meta.get("source", "unknown"))
        page = meta.get("page", None)
        tag = f"(source={src}" + (f", page={page}" if page is not None else "") + ")"
        selected.append(d.page_content.strip() + f"\n{tag}")
    return "\n---\n".join(selected)

# (옵션) 간단 키워드 기반 백업 검색
def pick_context_by_keyword(query: str, k: int = 4) -> str:
    tokens = re.findall(r"[A-Za-z가-힣0-9]+", query.lower())
    tokens = [t for t in tokens if len(t) > 1]
    scored = []
    for i, d in enumerate(chunks):
        txt = d.page_content.lower()
        score = sum(txt.count(t) for t in tokens)
        scored.append((score, i))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
    selected = []
    for _, idx in top:
        meta = chunks[idx].metadata or {}
        src = os.path.basename(meta.get("source", "unknown"))
        page = meta.get("page", None)
        tag = f"(source={src}" + (f", page={page}" if page is not None else "") + ")"
        selected.append(chunks[idx].page_content.strip() + f"\n{tag}")
    return "\n---\n".join(selected)

def get_context(query: str, k: int = 4) -> str:
    """우선 FAISS, 불가 시 키워드 방식."""
    ctx = pick_context_by_faiss(query, k=k)
    if not ctx:
        ctx = pick_context_by_keyword(query, k=k)
    return ctx

# ====================== Preview helpers (hide blocks) ======================
def strip_preview_blocks(text: str) -> str:
    """
    프리뷰에서 다음 블록을 제거:
      [컨텍스트] ... (끝까지)
      [작업] ... (다음 대괄호 헤더 또는 끝까지)
    """
    t = re.sub(r"\n?\[컨텍스트\][\s\S]*$", "", text)
    t = re.sub(r"\n?\[작업\][\s\S]*?(?=\n\[[^\]\n]+\]|\Z)", "", t)
    return t.rstrip()

def preview_messages_without_context(messages) -> str:
    rows = []
    for m in messages:
        role = getattr(m, "role", getattr(m, "type", "message"))
        content = strip_preview_blocks(m.content)
        rows.append(f"{role}: {content}")
    return "\n".join(rows)

# ============================== 1) PromptTemplate ==============================
print_section("1) PromptTemplate")

pt_template = """아래 컨텍스트만 근거로, 요청된 작업을 수행하세요.
- 한국어로 간결하게 답변
- 숫자/고유명사는 컨텍스트에 있는 것만 사용
- 마지막에 1줄 요약 추가
[작업]
{task}

[컨텍스트]
{context}
"""
pt = PromptTemplate.from_template(pt_template)

user_task = "SPRi AI Brief와 AI4Science 보고서를 바탕으로 'AI가 과학 연구 패러다임에 미친 영향'을 5가지 bullet로 요약해줘. 각 bullet은 20자 내외."
context_1 = get_context("AI4Science 연구 패러다임 SPRi AI Brief 정책 변화 과학 혁명", k=4)

formatted_1 = pt.format(task=user_task, context=context_1)   # 모델 입력(컨텍스트 포함)
preview_1 = strip_preview_blocks(formatted_1)                # 프리뷰(작업/컨텍스트 제거)

resp_text = None
if llm:
    resp_text = llm.invoke(formatted_1).content

show_triplet(
    user_input=f'task="{user_task}"',
    prompt_preview=preview_1,
    response_text=resp_text
)

# ============================ 2) ChatPromptTemplate ===========================
print_section("2) ChatPromptTemplate")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise Korean analyst. Use ONLY the provided context. "
     "Return JSON with keys: answer (string, Korean), sources (array of strings: 'filename:page'), "
     "and keywords (array of 3-6 important terms from context). No extra text."),
    ("human",
     "질문: {question}\n\n[컨텍스트]\n{context}")
])

user_question = "금융과 NLP 키워드 중 서로 연관된 개념 3쌍을 찾아 간단히 설명해줘."
context_2 = get_context("finance keywords nlp keywords 의미론 임베딩 투자 지표", k=4)

messages_2 = chat_prompt.format_messages(question=user_question, context=context_2)  # 모델 입력
preview_2 = preview_messages_without_context(messages_2)                              # 프리뷰

resp_text = None
if llm:
    resp_text = llm.invoke(messages_2).content

show_triplet(
    user_input=f'question="{user_question}"',
    prompt_preview=preview_2,
    response_text=resp_text
)

# =========================== 3) FewShotPromptTemplate =========================
print_section("3) FewShotPromptTemplate")

few_examples = [
    {
        "goal": "키워드 사전(텍스트)에서 투자/시장 관련 핵심 용어 2개를 찾아 정의를 한 줄씩 요약",
        "output": "Term: 지수(Index) - 시장 전반을 나타내는 대표 바스켓. (source=finance-keywords.txt)\n"
                  "Term: 변동성(Volatility) - 가격 변동 폭을 가리키는 지표. (source=finance-keywords.txt)"
    },
    {
        "goal": "AI 보고서(PDF)에서 연구 패러다임 변화를 한 줄 요약 2개",
        "output": "AI는 데이터주도 연구를 넘어 5번째 과학혁명의 동력. (source=IS-208...pdf)\n"
                  "가설-실험-분석 전주기에서 AI가 지능형 동반자로 기능. (source=IS-208...pdf)"
    },
]

example_prompt = LegacyPromptTemplate(
    input_variables=["goal", "output"],
    template="Goal:\n{goal}\n\nGood Output Example:\n{output}\n"
)

fewshot = FewShotPromptTemplate(
    examples=few_examples,
    example_prompt=example_prompt,
    suffix=("아래 컨텍스트만 보기로 하고, 동일한 형식으로 결과를 생성하세요.\n"
            "Goal:\n{goal}\n\n[컨텍스트]\n{context}\n\nYour Output:\n"),
    input_variables=["goal", "context"]
)

user_goal = "AI Brief와 AI4Science에서 '연구 단계별 AI 활용'을 2줄로 뽑아줘. 각 줄 끝에 출처 파일명을 괄호로 표시."
context_3 = get_context("가설 형성 실험 설계 데이터 수집 분석 단계 AI 역할", k=4)

fewshot_str = fewshot.format(goal=user_goal, context=context_3)   # 모델 입력
fewshot_preview = strip_preview_blocks(fewshot_str)               # 프리뷰

resp_text = None
if llm:
    resp_text = llm.invoke(fewshot_str).content

show_triplet(
    user_input=f'goal="{user_goal}"',
    prompt_preview=fewshot_preview,
    response_text=resp_text
)

from dotenv import load_dotenv
import os, os.path as op
from pathlib import Path
from operator import itemgetter

# 1) Loaders & Splitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 2) Embeddings, Vector Store, LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama  import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 3) LCEL + History
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

@contextlib.contextmanager
def log_step(name: str):
    t0 = time.perf_counter()
    logging.info(f"[START] {name}")
    try:
        yield
        dt = time.perf_counter() - t0
        logging.info(f"[OK] {name} ({dt:.2f}s)")
    except Exception as e:
        dt = time.perf_counter() - t0
        logging.exception(f"[FAIL] {name} after {dt:.2f}s: {e}")
        raise

# ============================== Bootstrap ==============================
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DOC_PATHS = [
    BASE_DIR / "finance-keywords.txt",
    BASE_DIR / "nlp-keywords.txt",
    BASE_DIR / "SPRi AI Brief_9월호_산업동향_0909_F.pdf",
    BASE_DIR / "IS-208 과학을 위한 AI(AI4Science) 연구의 패러다임을 바꾸다.pdf",
]

# ========================= Loaders & Splitter ==========================
def load_documents(paths):
    docs = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] Missing: {p}")
            continue
        try:
            if p.suffix.lower() == ".txt":
                docs += TextLoader(str(p), encoding="utf-8").load()
            elif p.suffix.lower() == ".pdf":
                docs += PyPDFLoader(str(p)).load()
            else:
                print(f"[WARN] Skip unsupported: {p}")
        except Exception as e:
            print(f"[WARN] Load fail {p}: {e}")
    if not docs:
        raise RuntimeError("No documents loaded. Check file paths.")
    return docs

docs = load_documents(DOC_PATHS)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n## ", "\n### ", "\n", " ", ""],
)
chunks = splitter.split_documents(docs)
if not chunks:
    raise RuntimeError("No chunks after splitting.")

# ===================== Embeddings & Vector Store (FAISS) =====================
# OpenAI Embeddings
# emb = OpenAIEmbeddings(model="text-embedding-3-small")

# Ollama Embeddings (BGEM3)
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# emb = OllamaEmbeddings(
#     model="bge-m3:567m",
#     base_url=OLLAMA_BASE_URL,
# )

# HuggingFace Embeddings
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL")
HF_DEVICE = os.getenv("HF_DEVICE")

emb = HuggingFaceEmbeddings(
    model_name=HF_EMBED_MODEL,
    model_kwargs={"device": HF_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

vs = FAISS.from_documents(chunks, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ================================ Prompt =====================================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise Korean analyst. Use ONLY the provided context to answer in Korean. "),
    MessagesPlaceholder("history"),
    ("human", "질문: {question}\n\n[컨텍스트]\n{context}")
])

# ============================ LCEL RAG Chain =================================
def format_docs(docs):
    parts = []
    for d in docs:
        meta = d.metadata or {}
        src = op.basename(meta.get("source", "unknown"))
        page = meta.get("page")
        tag = f"(source={src}" + (f", page={page}" if page is not None else "") + ")"
        parts.append(d.page_content.strip() + f"\n{tag}")
    return "\n---\n".join(parts)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

setup = RunnableParallel(
    context = itemgetter("question") | retriever | format_docs,
    question = itemgetter("question"),
    history  = itemgetter("history"),
)

rag_chain = setup | prompt | llm | StrOutputParser()

# ======================= History wrapper (session-based) ======================
_history_store = {}

def get_history(session_id: str):
    # 프로세스 메모리에 세션별 히스토리를 보관합니다.
    return _history_store.setdefault(session_id, ChatMessageHistory())

chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history=get_history,
    input_messages_key="question",   # 사용자 입력 키
    history_messages_key="history"   # 프롬프트의 MessagesPlaceholder 키
)

# ================================ REPL =======================================
if __name__ == "__main__":
    session_id = "local-user"  # 사용자/채널/탭 등으로 고정
    while True:
        try:
            q = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break
        if not q or q.lower() in {"q", "quit", "exit"}:
            print("[EXIT]")
            break

        # 같은 session_id로 호출하면 이전 대화가 자동 누적되어 사용됩니다.
        answer = chain.invoke(
            {"question": q},
            config={"configurable": {"session_id": session_id}}
        )

        print("\n답변>\n" + answer)

from dotenv import load_dotenv
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts import FewShotPromptTemplate, PromptTemplate as LegacyPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# ============================== Bootstrap ==============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HAS_KEY = bool(OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if HAS_KEY else None
if not HAS_KEY:
  print("[INFO] OPENAI_API_KEY가 없어 LLM 호출은 건너뜁니다. 템플릿 결과만 출력합니다.")

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

def load_documents(paths):
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

# =================== Simple keyword-based context picker ===================
def get_context(query: str, k: int = 4):
  tokens = re.findall(r"[A-Za-z가-힣0-9]+", query.lower())
  tokens = [t for t in tokens if len(t) > 1]
  scores = []
  for i, d in enumerate(chunks):
    txt = d.page_content.lower()
    score = sum(txt.count(t) for t in tokens)
    # 간단한 점수만 사용
    scores.append((score, i))
  top = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
  selected = []
  for _, idx in top:
    meta = chunks[idx].metadata
    src = os.path.basename(meta.get("source", "unknown"))
    page = meta.get("page", None)
    tag = f"(source={src}" + (f", page={page}" if page is not None else "") + ")"
    selected.append(chunks[idx].page_content.strip() + f"\n{tag}")
  return "\n---\n".join(selected) if selected else ""

# ====================== Preview helpers ======================
def strip_context_block(text: str) -> str:
  """
  문자열에서 [컨텍스트] 헤더와 그 이후 전체를 제거.
  - PromptTemplate 미리보기에서 사용
  """
  t = re.sub(r"\n?\[컨텍스트\][\s\S]*$", "", text)
  t = re.sub(r"\n?\[작업\][\s\S]*?(?=\n\[[^\]\n]+\]|\Z)", "", t)

  return t.rstrip()

def preview_messages_without_context(messages) -> str:
  """
  ChatPromptTemplate 미리보기:
  각 메시지 content에서 [컨텍스트] 블록 제거 후 직렬화.
  """
  rows = []
  for m in messages:
    role = getattr(m, "role", getattr(m, "type", "message"))
    content = strip_context_block(m.content)
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

# 모델 입력
formatted_1 = pt.format(task=user_task, context=context_1)
# 출력 프리뷰
preview_1 = strip_context_block(formatted_1)

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

# 모델 입력
messages_2 = chat_prompt.format_messages(question=user_question, context=context_2)
# 출력 프리뷰
preview_2 = preview_messages_without_context(messages_2)

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

# 모델 입력
fewshot_str = fewshot.format(goal=user_goal, context=context_3)
# 출력 프리뷰
fewshot_preview = strip_context_block(fewshot_str)

resp_text = None
if llm:
  resp_text = llm.invoke(fewshot_str).content

show_triplet(
  user_input=f'goal="{user_goal}"',
  prompt_preview=fewshot_preview,
  response_text=resp_text
)

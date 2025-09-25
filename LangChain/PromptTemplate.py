from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HAS_KEY = bool(OPENAI_API_KEY)

def print_section(title):
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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if HAS_KEY else None
if not HAS_KEY:
    print("[INFO] OPENAI_API_KEY가 없어 LLM 호출은 건너뜁니다. 템플릿 결과만 출력합니다.")

# ------------------------------------------------------------------------------
# 1) PromptTemplate(평균 계산)
from langchain_core.prompts import PromptTemplate

print_section("1) PromptTemplate")

# 숫자 리스트 평균 계산(소수점 둘째 자리, 숫자만 출력)
template = "다음 숫자들의 평균을 소수점 둘째 자리까지 계산하고, 설명 없이 숫자만 출력하세요: {numbers}"
prompt = PromptTemplate.from_template(template)

user_numbers = "10, 20, 31"  # 사용자 입력
formatted = prompt.format(numbers=user_numbers)

resp_text = None
if llm:
    resp_text = llm.invoke(formatted).content

show_triplet(
    user_input=f'numbers="{user_numbers}"',
    prompt_preview=formatted,
    response_text=resp_text
)

# ------------------------------------------------------------------------------
# 2) ChatPromptTemplate(리뷰 분석 → JSON)
from langchain_core.prompts import ChatPromptTemplate

print_section("2) ChatPromptTemplate")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Return ONLY JSON with keys: "
     "sentiment(one of positive|neutral|negative), keywords(array, 2-5), summary(Korean, <=20 words)."),
    ("human", "다음 고객 리뷰를 분석해 주세요: {review}")
])

user_review = "배송은 빨랐지만 포장이 허술해서 박스가 일부 찌그러졌어요. 제품은 정상 작동합니다."  # 사용자 입력
messages = chat_prompt.format_messages(review=user_review)

preview = "\n".join([f"{getattr(m,'role','') or getattr(m,'type','')}: {m.content}" for m in messages])

resp_text = None
if llm:
    resp_text = llm.invoke(messages).content

show_triplet(
    user_input=f'review="{user_review}"',
    prompt_preview=preview,
    response_text=resp_text
)

# ------------------------------------------------------------------------------
# 3) FewShotPromptTemplate(동의어 찾기)
from langchain.prompts import FewShotPromptTemplate, PromptTemplate as LegacyPromptTemplate

print_section("3) FewShotPromptTemplate")

examples = [
    {"word": "rapid", "synonym": "fast"},
    {"word": "big",   "synonym": "large"},
]

example_prompt = LegacyPromptTemplate(
    input_variables=["word", "synonym"],
    template="Word: {word}\nSynonym: {synonym}\n"
)

fewshot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Word: {input}\nSynonym:",
    input_variables=["input"]
)

user_word = "difficult"  # 사용자 입력
fewshot_str = fewshot_prompt.format(input=user_word)

resp_text = None
if llm:
    resp_text = llm.invoke(fewshot_str).content

show_triplet(
    user_input=f'input="{user_word}"',
    prompt_preview=fewshot_str,
    response_text=resp_text
)

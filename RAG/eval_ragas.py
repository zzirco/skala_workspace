# eval_ragas.py
"""
RAGAS로 Advanced RAG 파이프라인을 평가합니다.
Dataset 포맷: JSONL (eval_dataset.jsonl)
각 줄 예시:
{"question": "저작권 침해 손해배상 산정 기준은?",
 "gold_contexts": ["...정답 근거 문장 A...", "...근거 문장 B..."],  # 선택(없어도 일부 지표는 가능)
 "reference_answer": "판례상 손해배상 산정은 ..."                    # 선택
}

실행:
(venv) > python eval_ragas.py --data eval_dataset.jsonl --limit 200
"""

import argparse, json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rag_advanced import ask_advanced  # 기존 파이프라인 호출

# RAGAS
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

def load_jsonl(path: str, limit: int | None = None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="eval_dataset.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="평가 샘플 상한")
    args = ap.parse_args()

    ds = load_jsonl(args.data, args.limit)
    if not ds:
        raise ValueError("데이터셋이 비어있습니다. eval_dataset.jsonl을 확인하세요.")

    predictions, references = [], []

    for ex in ds:
        q = ex["question"]
        out = ask_advanced(q, history=ex.get("history", ""))  # 우리 파이프라인 호출

        # 예측: answer + 모델이 사용한 contexts (압축본 기준)
        ctxs = []
        for d in out.get("docs_compressed", []):
            ctxs.append(d.page_content)

        predictions.append({
            "question": q,
            "answer": out.get("answer", ""),
            "contexts": ctxs,
        })

        # 레퍼런스(선택 항목 있어도/없어도 됨)
        references.append({
            "question": q,
            "contexts": ex.get("gold_contexts", []),     # 없으면 빈 리스트
            "answer": ex.get("reference_answer", ""),    # 없으면 빈 문자열
        })

    # RAGAS 평가 실행
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(predictions=predictions, references=references, metrics=metrics)

    # 결과 출력
    print("\n=== RAGAS Results (mean) ===")
    for k, v in result.items():
        # result는 dict-like; .get 로 평균값 접근이 가능
        # 일부 버전은 result[k].score 같은 속성일 수 있으니 안전하게 처리
        try:
            score = getattr(v, "score", v)
        except Exception:
            score = v
        print(f"{k:20s}: {score:.4f}")

    # 상세 리포트 저장
    out_path = Path("ragas_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, default=lambda o: getattr(o, "__dict__", str(o)))
    print(f"\n📄 Saved detailed report -> {out_path.resolve()}")

if __name__ == "__main__":
    main()

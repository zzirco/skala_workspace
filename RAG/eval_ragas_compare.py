# eval_ragas_compare.py
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def call_pipeline(pipeline: str, question: str, history: str = "") -> Dict[str, Any]:
    if pipeline == "naive":
        from rag_naive import ask_naive
        return ask_naive(question, history)
    else:
        from rag_advanced import ask_advanced
        return ask_advanced(question, history)


def extract_contexts(out: Dict[str, Any]) -> List[str]:
    docs = out.get("docs_compressed") or out.get("docs_ranked") or out.get("docs") or []
    ctxs: List[str] = []
    for d in docs:
        txt = getattr(d, "page_content", None)
        if txt is None and isinstance(d, dict):
            txt = d.get("page_content") or d.get("content") or d.get("text")
        if txt:
            ctxs.append(str(txt))
    return ctxs


def build_dataset(rows: List[Dict[str, Any]], pipeline: str, use_history: bool = False) -> Dataset:
    data: List[Dict[str, Any]] = []
    for ex in tqdm(rows, desc=f"Pipeline={pipeline}"):
        q = (ex.get("question") or "").strip()
        if not q:
            continue
        history = (ex.get("history") or "") if use_history else ""
        out = call_pipeline(pipeline, q, history)

        ans = (out.get("answer") or "").strip()
        ctxs = extract_contexts(out)

        # 선택 필드들(없으면 빈 값)
        gold_ctxs = ex.get("gold_contexts") or []
        reference_answer = (ex.get("reference_answer") or "").strip()

        data.append({
            "question": q,
            "answer": ans,
            "contexts": ctxs,
            # 최신 ragas에서 일부 지표가 요구하는 컬럼들
            "ground_truths": gold_ctxs,   # 있으면 좋음(맥락 관련 지표 강화)
            "reference": reference_answer # 없으면 빈 문자열 → 아래에서 지표 자동 조정
        })

    if not data:
        raise ValueError("평가에 사용할 row가 없습니다. eval_dataset.jsonl을 확인하세요.")
    return Dataset.from_list(data)


def run_ragas(ds: Dataset) -> Dict[str, float]:
    """
    RAGAS 평가 실행.
    - 버전 차이에 따라 결과 접근 방식이 달라 예외 없이 점수만 추출하도록 방어적으로 처리
    """
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # reference가 없으면 context_*는 제외 (이전 단계에서 이미 처리하셨다면 생략 가능)
    needs_ref = {"context_precision", "context_recall"}
    if "reference" not in ds.column_names or not any((ds["reference"])):
        metrics = [m for m in metrics if getattr(m, "name", "") not in needs_ref]

    result = ragas_evaluate(ds, metrics=metrics)

    # --- 다양한 버전에 대응하여 점수 추출 ---
    scores: Dict[str, float] = {}

    # 1) 가장 호환성 높은 방식: to_pandas() 사용
    try:
        df = result.to_pandas()  # columns 예: ["metric", "score", ...]
        for _, row in df.iterrows():
            mname = str(row.get("metric"))
            mscore = row.get("score")
            if mname and mscore is not None:
                scores[mname] = float(mscore)
        if scores:
            return scores
    except Exception:
        pass

    # 2) result.scores(dict-like) 또는 metric별 객체(score 속성 보유)
    try:
        if hasattr(result, "scores"):
            rs = getattr(result, "scores")
            if isinstance(rs, dict):
                for k, v in rs.items():
                    # v가 float이거나 객체(score 속성)일 수 있음
                    if hasattr(v, "score"):
                        scores[str(k)] = float(v.score)
                    else:
                        scores[str(k)] = float(v)
                if scores:
                    return scores
    except Exception:
        pass

    # 3) result[...] 인덱싱 가능한 경우 (리스트/객체 혼재)
    for m in metrics:
        name = getattr(m, "name", str(m))
        try:
            item = result[name]
            if hasattr(item, "score"):
                scores[name] = float(item.score)
            elif isinstance(item, (int, float)):
                scores[name] = float(item)
            else:
                # 리스트 등일 경우 평균 등 임시 처리
                try:
                    from statistics import mean
                    scores[name] = float(mean(item))
                except Exception:
                    continue
        except Exception:
            continue

    return scores



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="평가용 JSONL (eval_dataset.jsonl)")
    ap.add_argument("--pipelines", nargs="+", default=["advanced", "naive"], choices=["advanced", "naive"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--use_history", action="store_true")
    ap.add_argument("--out", default="ragas_compare_results.json")
    args = ap.parse_args()

    rows = load_jsonl(args.data, args.limit)
    if not rows:
        print(f"[!] 데이터가 비어있습니다: {args.data}")
        return

    all_results: Dict[str, Any] = {"dataset_path": args.data, "count": len(rows), "pipelines": {}}

    for p in args.pipelines:
        print(f"\n=== Evaluate pipeline: {p} ===")
        ds = build_dataset(rows, pipeline=p, use_history=args.use_history)
        scores = run_ragas(ds)
        all_results["pipelines"][p] = {
            "metrics": scores,
            "sample": ds.select(range(min(3, len(ds)))).to_list(),
        }
        print(f"[{p}] samples={len(ds)}")
        for k, v in scores.items():
            print(f"  - {k}: {v:.4f}")

    if len(args.pipelines) > 1:
        print("\n=== Summary (side-by-side) ===")
        keys = sorted({k for p in args.pipelines for k in all_results["pipelines"][p]["metrics"].keys()})
        header = "metric".ljust(24) + " ".join([p.ljust(12) for p in args.pipelines])
        print(header)
        print("-" * len(header))
        for k in keys:
            row = k.ljust(24)
            for p in args.pipelines:
                row += f"{all_results['pipelines'][p]['metrics'].get(k,0.0):.4f}".ljust(12)
            print(row)

    Path(args.out).write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n✅ Saved: {args.out}")


if __name__ == "__main__":
    main()

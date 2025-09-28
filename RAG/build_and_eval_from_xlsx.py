# build_and_eval_from_xlsx.py
"""
엑셀/CSV(Q&A) -> eval_dataset.jsonl 생성 -> RAGAS 평가(advanced/naive 비교)

사용 예:
(venv)> python build_and_eval_from_xlsx.py --file "1. 대한법률구조공단_Q&A.xlsx" --pipelines advanced naive --limit 200

옵션:
  --file            : 입력 파일 경로(.xlsx 또는 .csv)
  --qcol / --acol   : 질문/정답 컬럼명 직접 지정(자동 탐지 실패시)
  --limit           : 상위 N개만 사용(생성 및 평가 모두에 적용)
  --out-dataset     : 생성할 JSONL 경로 (기본: eval_dataset.jsonl)
  --out-results     : 평가 결과 JSON 경로 (기본: ragas_compare_results.json)
  --pipelines       : 평가 파이프라인 목록 (advanced|naive)
  --use_history     : (옵션) history 열이 있을 경우 반영

전제:
- 프로젝트 내에 rag_advanced.py, rag_naive.py가 존재, 동작
- OPENAI_API_KEY 환경 변수 설정(ragas가 LLM 호출)
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv

# RAGAS
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

load_dotenv()

# ------------------------ 유틸 ------------------------
Q_CANDIDATES = ["질문", "문의", "Q", "question", "Question", "제목", "상담질문", "질의"]
A_CANDIDATES = ["답변", "A", "answer", "Answer", "응답", "상담답변", "내용", "본문"]

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        # BOM/일반 모두 대응
        for enc in ["utf-8-sig", "utf-8", "cp949"]:
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception:
                continue
        # 마지막 시도: 기본
        return pd.read_csv(p)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. .xlsx 또는 .csv를 사용하세요.")

def detect_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    # 완전일치 우선, 소문자 비교 보조
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def safe_str(x) -> str:
    try:
        return (str(x).strip()) if pd.notna(x) else ""
    except Exception:
        return ""

def build_eval_dataset_jsonl(
    df: pd.DataFrame,
    out_path: str,
    qcol: Optional[str] = None,
    acol: Optional[str] = None,
    limit: Optional[int] = None,
) -> int:
    cols = list(df.columns)
    q_col = qcol or detect_column(cols, Q_CANDIDATES)
    a_col = acol or detect_column(cols, A_CANDIDATES)

    if not q_col:
        raise ValueError(f"질문 컬럼을 찾지 못했습니다. 실제 컬럼들: {cols}\n--qcol 옵션으로 지정하세요.")
    # 정답(레퍼런스)은 없어도 생성 가능 (컨텍스트 지표는 비활성화)
    if a_col is None:
        print("[i] 정답 컬럼이 없어 최소 포맷(question-only)으로 생성합니다.")

    rows = df.to_dict(orient="records")
    if limit:
        rows = rows[:limit]

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            q = safe_str(r.get(q_col, ""))
            if not q:
                continue
            ref = safe_str(r.get(a_col, "")) if a_col else ""
            item = {
                "question": q,
                # RAGAS context_* 지표 활성화를 원하면 reference_answer를 채우세요.
                "reference_answer": ref,
                # gold_contexts는 없으면 빈 리스트
                "gold_contexts": []
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n

# ------------------- 파이프라인 호출 & 평가 -------------------
def call_pipeline(pipeline: str, question: str, history: str = "") -> Dict[str, Any]:
    if pipeline == "naive":
        from rag_naive import ask_naive
        return ask_naive(question, history)
    else:
        from rag_advanced import ask_advanced
        return ask_advanced(question, history)

def extract_contexts_lite(out: Dict[str, Any], k: int = 4, max_chars: int = 800) -> List[str]:
    docs = out.get("docs_compressed") or out.get("docs_ranked") or out.get("docs") or []
    ctxs: List[str] = []
    for d in docs[:k]:
        txt = getattr(d, "page_content", None)
        if txt is None and isinstance(d, dict):
            txt = d.get("page_content") or d.get("content") or d.get("text")
        if txt:
            ctxs.append(str(txt)[:max_chars])
    return ctxs

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

def build_ds_for_ragas(rows: List[Dict[str, Any]], pipeline: str, use_history: bool = False) -> Dataset:
    data: List[Dict[str, Any]] = []
    for ex in rows:
        q = safe_str(ex.get("question", ""))
        if not q:
            continue
        history = safe_str(ex.get("history", "")) if use_history else ""
        out = call_pipeline(pipeline, q, history)
        ans = safe_str(out.get("answer", ""))
        ctxs = extract_contexts_lite(out)
        ref_ans = safe_str(ex.get("reference_answer", ""))  # 없으면 ""

        data.append({
            "question": q,
            "answer": ans,
            "contexts": ctxs,
            "ground_truths": ex.get("gold_contexts", []),  # []
            "reference": ref_ans,
        })
    if not data:
        raise ValueError("평가에 사용할 데이터가 비었습니다.")
    return Dataset.from_list(data)

def run_ragas(ds: Dataset) -> Dict[str, float]:
    # reference가 없으면 context_* 제외
    has_ref = "reference" in ds.column_names and any(bool((x or "").strip()) for x in ds["reference"])
    metrics = [faithfulness, answer_relevancy]
    if has_ref:
        metrics += [context_precision, context_recall]

    result = ragas_evaluate(ds, metrics=metrics)

    # 버전별 결과 추출 방어 처리
    scores: Dict[str, float] = {}
    # 1) pandas
    try:
        df = result.to_pandas()
        for _, row in df.iterrows():
            m = str(row.get("metric"))
            s = row.get("score")
            if m and s is not None:
                scores[m] = float(s)
        if scores:
            return scores
    except Exception:
        pass
    # 2) scores dict-like
    try:
        if hasattr(result, "scores") and isinstance(result.scores, dict):
            for k, v in result.scores.items():
                val = float(getattr(v, "score", v))
                scores[str(k)] = val
            if scores:
                return scores
    except Exception:
        pass
    # 3) metric별 접근
    for m in metrics:
        name = getattr(m, "name", str(m))
        try:
            item = result[name]
            if hasattr(item, "score"):
                scores[name] = float(item.score)
        except Exception:
            continue
    return scores

# ------------------------ 메인 ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="입력 파일(.xlsx 또는 .csv)")
    ap.add_argument("--qcol", default=None, help="질문 컬럼명 직접 지정(자동 탐지 실패시)")
    ap.add_argument("--acol", default=None, help="정답 컬럼명(있으면 context_* 지표 활성화)")
    ap.add_argument("--limit", type=int, default=None, help="상위 N개만 사용")
    ap.add_argument("--out-dataset", default="eval_dataset.jsonl", help="생성할 JSONL 경로")
    ap.add_argument("--pipelines", nargs="+", default=["advanced", "naive"], choices=["advanced", "naive"])
    ap.add_argument("--use_history", action="store_true", help="입력 파일에 history 열이 있으면 반영")
    ap.add_argument("--out-results", default="ragas_compare_results.json", help="RAGAS 결과 저장 경로")
    args = ap.parse_args()

    # 1) 파일 읽기 & 데이터셋 생성
    df = read_table(args.file)
    print(f"Loaded rows={len(df)} cols={list(df.columns)}")
    n = build_eval_dataset_jsonl(
        df,
        out_path=args.out_dataset,
        qcol=args.qcol,
        acol=args.acol,
        limit=args.limit
    )
    print(f"✅ 생성 완료: {args.out_dataset} (샘플 {n}개)")

    # 2) 생성된 jsonl 로드
    rows = load_jsonl(args.out_dataset, args.limit)
    if not rows:
        print("[!] 생성된 평가셋이 비어있습니다.")
        return

    # 3) 파이프라인별 평가
    all_results: Dict[str, Any] = {"dataset_path": args.out_dataset, "count": len(rows), "pipelines": {}}

    for p in args.pipelines:
        print(f"\n=== Evaluate pipeline: {p} ===")
        ds = build_ds_for_ragas(rows, pipeline=p, use_history=args.use_history)
        scores = run_ragas(ds)
        all_results["pipelines"][p] = {
            "metrics": scores,
            "sample": ds.select(range(min(3, len(ds)))).to_list(),
        }
        print(f"[{p}] samples={len(ds)}")
        for k, v in scores.items():
            print(f"  - {k}: {v:.4f}")

    # 4) 비교 표
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

    Path(args.out_results).write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Saved results: {args.out_results}")


if __name__ == "__main__":
    main()

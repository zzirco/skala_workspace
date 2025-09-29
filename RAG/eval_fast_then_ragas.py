# eval_fast_then_ragas.py
"""
엑셀/CSV -> 파이프라인(advanced/naive) 호출 -> FastEval(로컬) -> 대표샘플만 RAGAS(소량)
- LLM 호출 실패/빈 샘플 자동 필터링
- RAGAS는 faithfulness/answer_relevancy 중심(레퍼런스 있으면 context_* 자동 활성)

사용 예:
(venv) > python eval_fast_then_ragas.py --file "1. 대한법률구조공단_Q&A.xlsx" --pipelines advanced naive --limit 200 --ragas_max 50

필수:
- 프로젝트 루트에 rag_advanced.py / rag_naive.py 존재
- .env에 OPENAI_API_KEY (RAGAS에서만 필요, FastEval은 불필요)
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Fast local metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# RAGAS
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


# ------------------------ 입출력/전처리 ------------------------
Q_CANDIDATES = ["질문", "문의", "Q", "question", "Question", "제목", "상담질문", "질의"]
A_CANDIDATES = ["답변", "A", "answer", "Answer", "응답", "상담답변", "내용", "본문"]

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        for enc in ["utf-8-sig", "utf-8", "cp949"]:
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(p)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. .xlsx 또는 .csv를 사용하세요.")

def detect_column(cols: List[str], candidates: List[str]) -> Optional[str]:
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


# ------------------------ 파이프라인 호출 ------------------------
def call_pipeline(pipeline: str, question: str, history: str = "") -> Dict[str, Any]:
    if pipeline == "naive":
        from rag_naive import ask_naive
        return ask_naive(question, history)
    else:
        from rag_advanced import ask_advanced
        return ask_advanced(question, history)

def extract_contexts(out: Dict[str, Any], k: int = 4, max_chars: int = 800) -> List[str]:
    docs = out.get("docs_compressed") or out.get("docs_ranked") or out.get("docs") or []
    ctxs: List[str] = []
    for d in docs[:k]:
        txt = getattr(d, "page_content", None)
        if txt is None and isinstance(d, dict):
            txt = d.get("page_content") or d.get("content") or d.get("text")
        if txt:
            ctxs.append(str(txt)[:max_chars])
    return ctxs


# ------------------------ FastEval (로컬) ------------------------
def tfidf_cosine(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if not a.strip() or not b.strip():
        return 0.0
    vect = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vect.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0,0])

def fast_eval_row(q: str, ans: str, ctxs: List[str]) -> Dict[str, float]:
    ctx_concat = " ".join(ctxs)
    return {
        "fast_answer_relevancy": tfidf_cosine(q, ans),      # Q vs A
        "fast_context_alignment": tfidf_cosine(ans, ctx_concat),  # A vs C
        "ctx_count": float(len(ctxs)),
        "ans_len": float(len(ans)),
    }

def choose_representative_indices(scores: List[Dict[str, float]], max_n: int = 60) -> List[int]:
    """
    fast_answer_relevancy 기반으로 0~1 구간을 균등 버킷팅하여 대표 샘플 추출.
    """
    if not scores:
        return []
    vals = [s["fast_answer_relevancy"] for s in scores]
    order = np.argsort(vals)  # 낮은~높은
    if len(order) <= max_n:
        return order.tolist()
    # 균등 샘플링
    idxs = np.linspace(0, len(order)-1, num=max_n, dtype=int)
    idxs = [i for i in idxs if i < len(order)]  # 잘못된 인덱스 제거
    return [int(order[i]) for i in idxs]


# ------------------------ RAGAS용 Dataset 구성 ------------------------
def build_dataset_for_pipeline(
    rows: List[Dict[str, Any]],
    pipeline: str,
    use_history: bool = False,
    cache_path: Optional[Path] = None,
) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """
    rows: [{"question":..., "reference_answer":..., "gold_contexts":[...]}]
    cache_path: predictions 캐시(jsonl). 있으면 불러오고, 없으면 생성 후 저장.
    """
    preds: List[Dict[str, Any]] = []

    # 캐시가 있으면 재사용
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                preds.append(json.loads(line))
        print(f"[cache] loaded predictions: {len(preds)}")
    else:
        for ex in rows:
            q = safe_str(ex.get("question", ""))
            if not q:
                continue
            history = safe_str(ex.get("history", "")) if use_history else ""
            out = call_pipeline(pipeline, q, history)

            ans = safe_str(out.get("answer", ""))
            ctxs = extract_contexts(out)

            preds.append({
                "question": q,
                "answer": ans,
                "contexts": ctxs,
                "reference": safe_str(ex.get("reference_answer", "")),
                "gold_contexts": ex.get("gold_contexts", []),
            })
        if cache_path:
            with cache_path.open("w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"[cache] saved predictions: {cache_path}")

    # FastEval 점수 계산 + 실패샘플 제거
    clean: List[Dict[str, Any]] = []
    fast_scores: List[Dict[str, float]] = []
    dropped = {"empty_answer":0, "empty_ctx":0}
    for p in preds:
        if not p["answer"]:
            dropped["empty_answer"] += 1
            continue
        if not p["contexts"]:
            dropped["empty_ctx"] += 1
            continue
        fe = fast_eval_row(p["question"], p["answer"], p["contexts"])
        fast_scores.append(fe)
        p["_fast"] = fe
        clean.append(p)

    if dropped["empty_answer"] or dropped["empty_ctx"]:
        print(f"[i] Dropped (empty): answer={dropped['empty_answer']}, contexts={dropped['empty_ctx']}")
    if not clean:
        raise ValueError("유효한 예측 샘플이 없습니다(빈 답변/컨텍스트).")

    ds = Dataset.from_list(clean)
    return ds, fast_scores


def to_ragas_subset(ds: Dataset, fast_scores: List[Dict[str, float]], max_n: int) -> Dataset:
    idxs = choose_representative_indices(fast_scores, max_n=max_n)
    subset = ds.select(idxs) if idxs else ds
    print(f"[subset] RAGAS on {len(subset)} / {len(ds)} samples")
    return subset


def run_ragas(ds: Dataset) -> Dict[str, float]:
    has_ref = "reference" in ds.column_names and any((x or "").strip() for x in ds["reference"])
    metrics = [faithfulness, answer_relevancy] + ([context_precision, context_recall] if has_ref else [])
    result = ragas_evaluate(ds, metrics=metrics)

    # 결과 추출(버전 호환)
    scores: Dict[str, float] = {}
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
    if hasattr(result, "scores") and isinstance(result.scores, dict):
        for k, v in result.scores.items():
            scores[str(k)] = float(getattr(v, "score", v))
    return scores


# ------------------------ 메인 ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="엑셀/CSV 파일 경로")
    ap.add_argument("--qcol", default=None)
    ap.add_argument("--acol", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pipelines", nargs="+", default=["advanced", "naive"], choices=["advanced", "naive"])
    ap.add_argument("--use_history", action="store_true")
    ap.add_argument("--ragas_max", type=int, default=50, help="RAGAS로 정밀평가할 최대 샘플 수")
    ap.add_argument("--cache_dir", default="./.eval_cache", help="파이프라인 출력 캐시 디렉토리")
    ap.add_argument("--out", default="fast_then_ragas_results.json")
    args = ap.parse_args()

    # 1) 입력 로딩
    df = read_table(args.file)
    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[:args.limit]

    # 컬럼 감지
    cols = list(df.columns)
    q_col = args.qcol or detect_column(cols, Q_CANDIDATES)
    a_col = args.acol or detect_column(cols, A_CANDIDATES)
    if not q_col:
        raise ValueError(f"질문 컬럼을 찾지 못했습니다. 실제 컬럼: {cols}. --qcol로 지정하세요.")
    print(f"Loaded rows={len(rows)} cols={cols} (Q='{q_col}' A='{a_col or 'N/A'}')")

    # eval rows 포맷 정규화
    eval_rows: List[Dict[str, Any]] = []
    for r in rows:
        eval_rows.append({
            "question": safe_str(r.get(q_col, "")),
            "reference_answer": safe_str(r.get(a_col, "")) if a_col else "",
            "gold_contexts": []
        })

    # 2) 파이프라인별 수행 + FastEval + 대표 샘플 추출 + RAGAS(소량)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Any] = {"file": args.file, "count": len(eval_rows), "pipelines": {}}

    for p in args.pipelines:
        print(f"\n=== Pipeline: {p} ===")
        cache_path = cache_dir / f"pred_{p}.jsonl"
        ds, fast_scores = build_dataset_for_pipeline(
            eval_rows, pipeline=p, use_history=args.use_history, cache_path=cache_path
        )

        # FastEval 요약(평균)
        fe_keys = ["fast_answer_relevancy", "fast_context_alignment"]
        fe_means = {k: float(np.mean([s[k] for s in fast_scores])) for k in fe_keys}
        print(f"[FastEval mean] {fe_means}")

        # 대표 샘플만 추출해 RAGAS
        ds_sub = to_ragas_subset(ds, fast_scores, max_n=min(args.ragas_max, len(ds)))
        ragas_scores = run_ragas(ds_sub)

        all_results["pipelines"][p] = {
            "fast_eval_mean": fe_means,
            "ragas_metrics": ragas_scores,
            "ragas_samples": len(ds_sub),
            "total_valid_samples": len(ds),
        }

        print(f"[RAGAS] samples={len(ds_sub)} metrics={ragas_scores}")

    Path(args.out).write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {args.out}")


if __name__ == "__main__":
    main()

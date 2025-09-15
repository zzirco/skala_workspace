# pjt-main.py
# ---------------------------------------------
# FastAPI: 센서 데이터 업로드 → 서버에서 EDA 수행 → 이미지/요약 생성
# → 전처리/모델 학습/성능평가(시각화) → LLM Markdown 리포트 생성 → HTML 프리뷰 제공
#
# 필요 패키지(예시):
#   pip install fastapi uvicorn pandas numpy matplotlib seaborn scipy python-dotenv pydantic markdown langchain-openai scikit-learn
#   (엑셀 파일 지원시: pip install openpyxl)
# 환경변수: .env 파일에 OPENAI_API_KEY=sk-... 설정
# ---------------------------------------------

import os
import io
import re
import json
import uuid
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")  # 서버 사이드 렌더링(디스플레이 없이)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import markdown as md

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

from starlette.concurrency import run_in_threadpool

# -------------------------
# 환경 변수 로드
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env file.")

# -------------------------
# 경로/앱 초기화
# -------------------------
IMAGE_DIR = os.path.abspath("./eda-images")
os.makedirs(IMAGE_DIR, exist_ok=True)

ML_IMAGE_DIR = os.path.abspath("./ml-images")
os.makedirs(ML_IMAGE_DIR, exist_ok=True)

ARTIFACT_DIR = os.path.abspath("./eda-artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

app = FastAPI(
    title="Sensor EDA + ML Report API",
    description="센서 데이터 업로드 → 서버에서 EDA/전처리/모델학습/평가 실행 → 시각화 및 리포트 생성",
    version="2.0.0",
)

# 이미지 정적 서빙
app.mount("/eda-images", StaticFiles(directory=IMAGE_DIR), name="eda-images")
app.mount("/ml-images", StaticFiles(directory=ML_IMAGE_DIR), name="ml-images")

# 최근 실행 결과(데모용, 실제 운영에선 DB/S3 등 사용 권장)
LAST_RUN_SUMMARY: Optional[Dict[str, Any]] = None
LAST_RUN_FIGURES: Optional[List[Dict[str, Any]]] = None
LAST_RUN_ID: Optional[str] = None


# -------------------------
# 유틸 함수
# -------------------------
def _save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def _guess_time_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in ("time", "timestamp", "date", "datetime")]
    for c in candidates + list(df.columns):
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None

def _read_any(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath)
    elif ext in [".parquet"]:
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def _ensure_numeric(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def _resample_if_needed(s: pd.Series, rule: str = "1min") -> pd.Series:
    try:
        freq = pd.infer_freq(s.index[:20])
    except Exception:
        freq = None
    if isinstance(s.index, pd.DatetimeIndex) and freq is None:
        return s.resample(rule).mean()
    return s

def _prefix_image_urls(markdown_text: str, base_url: str) -> str:
    return re.sub(
        r'!\[([^\]]*)\]\(/eda-images/([^)]+)\)',
        rf'![\1]({base_url}eda-images/\2)',
        markdown_text
    )

def _detect_stage_columns(frame: pd.DataFrame) -> Dict[str, List[str]]:
    # stage{n}_ 접두어 자동 탐지
    pattern = re.compile(r"^stage(\d+)_")
    stage_nums = set()
    for c in frame.columns:
        m = pattern.match(c)
        if m:
            stage_nums.add(int(m.group(1)))
    stages = {}
    for n in sorted(stage_nums):
        cols = [c for c in frame.columns if c.startswith(f"stage{n}_")]
        if cols:
            stages[f"stage{n}"] = cols
    return stages

# -------------------------
# EDA 파이프라인 (Stage별)
# -------------------------
def run_eda_pipeline(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
    top_n_sensors: int = 6,
    prefix: str = "",
) -> (Dict[str, Any], List[Dict[str, Any]]):
    """
    반환:
      - summary(dict): overall + per_stage 요약
      - fig_manifest(list): [{name,title,url,stage}], stage ∈ {"overall","stage1",...}
    """

    logger.info("EDA start")

    # ---------- 시간축 설정 ----------
    if time_col is None:
        time_col = _guess_time_col(df)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col)
        df = df.set_index(time_col)
    else:
        df = df.reset_index().rename(columns={"index": "row_index"}).set_index("row_index")

    # 그룹 컬럼 기본값 자동 추정
    if group_cols is None:
        group_cols = [c for c in df.columns if str(c).lower() in ("tool", "chamber", "lot", "recipe")]

    fig_manifest: List[Dict[str, Any]] = []

    def _add_fig(figs: List[Dict[str, Any]], fig_name: str, title: str, stage: Optional[str]):
        fig_path = os.path.join(IMAGE_DIR, f"{prefix}{fig_name}.png")
        _save_fig(fig_path)
        stage_label = stage if stage is not None else "overall"
        figs.append({
            "name": fig_name,
            "title": title,
            "url": f"/eda-images/{prefix}{fig_name}.png",
            "stage": stage_label,
        })

    def _run_single_eda(frame: pd.DataFrame, label: Optional[str]) -> (Dict[str, Any], List[Dict[str, Any]]):
        local_figs: List[Dict[str, Any]] = []
        exclude = group_cols or []
        num_cols = _ensure_numeric(frame, exclude)

        if len(num_cols) == 0:
            return {
                "n_rows": int(len(frame)),
                "n_numeric_sensors": 0,
                "top_std_sensors": [],
                "stats_table": [],
                "outlier_counts": {},
                "corr_top_pairs": [],
            }, local_figs

        desc = frame[num_cols].describe().T
        missing_rate = frame[num_cols].isna().mean().rename("missing_rate")
        stats_table = desc.join(missing_rate)

        corr = frame[num_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        zscores = frame[num_cols].apply(lambda s: np.abs(stats.zscore(s, nan_policy="omit")))
        outlier_counts = (zscores > 3).sum().sort_values(ascending=False)

        std_rank = frame[num_cols].std().sort_values(ascending=False)
        top_cols = list(std_rank.head(top_n_sensors).index)

        stage_slug = f"{label}_" if label else ""
        title_prefix = f"[{label}] " if label else ""

        # 1) 상관 히트맵
        if len(num_cols) >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
            plt.title(f"{title_prefix}Sensor Correlation (Pearson)")
            _add_fig(local_figs, f"{stage_slug}corr_heatmap", f"{title_prefix}Sensor Correlation (Pearson)", label)

        # 2) 시계열
        for col in top_cols:
            s = frame[col].copy()
            if isinstance(frame.index, pd.DatetimeIndex):
                s = _resample_if_needed(s)
            plt.figure(figsize=(11, 3.5))
            s.plot()
            plt.title(f"{title_prefix}Time Series - {col}")
            plt.xlabel("time"); plt.ylabel(col)
            _add_fig(local_figs, f"{stage_slug}ts_{col}", f"{title_prefix}Time Series - {col}", label)

        # 3) 분포/박스 (상위 4개)
        for col in top_cols[:min(4, len(top_cols))]:
            plt.figure(figsize=(6, 3.6))
            sns.histplot(frame[col].dropna(), bins=50, kde=True)
            plt.title(f"{title_prefix}Distribution - {col}")
            _add_fig(local_figs, f"{stage_slug}dist_{col}", f"{title_prefix}Distribution - {col}", label)

            plt.figure(figsize=(4, 3.6))
            sns.boxplot(x=frame[col].dropna())
            plt.title(f"{title_prefix}Boxplot - {col}")
            _add_fig(local_figs, f"{stage_slug}box_{col}", f"{title_prefix}Boxplot - {col}", label)

        # 4) SPC
        for col in top_cols[:min(4, len(top_cols))]:
            s = frame[col].dropna()
            if s.empty:
                continue
            mu, sigma = s.mean(), s.std()
            ucl, lcl = mu + 3*sigma, mu - 3*sigma
            s_plot = _resample_if_needed(s) if isinstance(frame.index, pd.DatetimeIndex) else s
            plt.figure(figsize=(11, 3.6))
            s_plot.plot(label=col)
            plt.axhline(mu, linestyle="--", label="Mean")
            plt.axhline(ucl, color="r", linestyle="--", label="UCL (+3σ)")
            plt.axhline(lcl, color="r", linestyle="--", label="LCL (-3σ)")
            plt.legend()
            plt.title(f"{title_prefix}SPC Chart - {col}")
            _add_fig(local_figs, f"{stage_slug}spc_{col}", f"{title_prefix}SPC Chart - {col}", label)

        # 5) 그룹별 박스(대표 1개 그룹)
        for g in (group_cols or [])[:1]:
            if g in frame.columns and len(top_cols) > 0:
                for col in top_cols[:min(3, len(top_cols))]:
                    plt.figure(figsize=(8, 3.8))
                    sns.boxplot(x=frame[g].astype(str), y=frame[col])
                    plt.title(f"{title_prefix}Box by {g} - {col}")
                    plt.xticks(rotation=20)
                    _add_fig(local_figs, f"{stage_slug}box_{g}_{col}", f"{title_prefix}Box by {g} - {col}", label)

        return {
            "n_rows": int(len(frame)),
            "n_numeric_sensors": int(len(num_cols)),
            "top_std_sensors": top_cols,
            "stats_table": stats_table.reset_index().rename(columns={"index": "sensor"}).to_dict(orient="records"),
            "outlier_counts": {k: int(v) for k, v in outlier_counts.to_dict().items()},
            "corr_top_pairs": [
                {"pair": (r, c), "corr": float(corr.loc[r, c])}
                for r in corr.index for c in corr.columns
                if r != c and abs(corr.loc[r, c]) > 0.8
            ][:15],
        }, local_figs

    # ---------- 전체(Overall) EDA ----------
    overall_summary, overall_figs = _run_single_eda(df, label=None)

    # ---------- Stage별 EDA ----------
    stage_map = _detect_stage_columns(df)  # {'stage1': [...], 'stage2': [...], ...}
    per_stage: Dict[str, Any] = {}
    stage_order: List[str] = list(stage_map.keys())

    for stage in stage_order:
        stage_cols = stage_map[stage]
        cols = stage_cols + [g for g in (group_cols or []) if g in df.columns]
        sub = df[cols].copy()
        if sub.empty or len(stage_cols) == 0:
            continue
        sub_summary, sub_figs = _run_single_eda(sub, label=stage)
        per_stage[stage] = sub_summary
        overall_figs.extend(sub_figs)

    # ---------- 최종 요약 & 저장 ----------
    run_id = prefix.rstrip("_")
    final_summary = {
        "time_col": time_col,
        "group_cols": group_cols,
        "stages": stage_order,
        "overall": overall_summary,
        "per_stage": per_stage,
    }

    with open(os.path.join(ARTIFACT_DIR, f"{run_id}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    return final_summary, overall_figs

# -------------------------
# 모델링 파이프라인 (전처리+학습+평가+시각화)
# -------------------------
def detect_target(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    휴리스틱으로 타겟/태스크 감지.
    return: (target_col, task_type)  task_type ∈ {"classification","regression",None}
    """
    # 후보 이름
    name_candidates = [
        "target", "label", "y", "pass", "fail", "is_fail", "defect", "bin", "yield", "quality"
    ]
    # 우선 이름으로 탐색
    for c in df.columns:
        lc = str(c).lower()
        if any(k == lc for k in name_candidates):
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
                return c, "regression"
            # 분류: 범주형이거나 유니크 수가 적음
            return c, "classification"

    # 이름이 없으면 "가장 오른쪽 컬럼" 후보로 시도(수치형/범주형 판단)
    last = df.columns[-1]
    s = df[last]
    if pd.api.types.is_numeric_dtype(s):
        # 값 종류가 적으면 분류, 많으면 회귀
        return last, ("classification" if s.nunique() <= 10 else "regression")
    else:
        return last, "classification"

# 교체: run_modeling_pipeline
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor

def run_modeling_pipeline(
    df: pd.DataFrame,
    time_col: Optional[str],
    group_cols: Optional[List[str]],
    prefix: str,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    model_type: str = "mlp",   # 'mlp' | 'rf'
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    감독학습:
      - y NaN 제거
      - 분류일 때 클래스 수 < 2면 Fallback
    비지도 Fallback:
      - IsolationForest 진단 플롯 생성(항상 그림 ≥1)
    """
    ml_figs: List[Dict[str, Any]] = []
    logger.info(f"[ML] start df={df.shape} time_col={time_col} target_col={target_col} task_type={task_type} model_type={model_type} prefix={prefix}")

    # ---------- 헬퍼: 수치 피처 선택 ----------
    def _numeric_feature_cols(frame: pd.DataFrame) -> List[str]:
        exclude = list(group_cols or [])
        if time_col and time_col in frame.columns:
            exclude.append(time_col)
        return [c for c in frame.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(frame[c])]

    # ---------- 비지도 Fallback ----------
    def _unsupervised(df_in: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        feat_cols = _numeric_feature_cols(df_in)
        if len(feat_cols) == 0:
            logger.warning("[ML] unsupervised skipped: no numeric features")
            return {"status":"unsupervised_skipped","reason":"no numeric features"}, []

        X = df_in[feat_cols].copy()
        imp = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imp.fit_transform(X), columns=feat_cols, index=df_in.index)

        iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
        iso.fit(X_imp)
        scores = -iso.score_samples(X_imp)  # 높을수록 이상치 경향

        # (1) 스코어 분포
        plt.figure(figsize=(5.5, 4.0))
        sns.histplot(scores, bins=60, kde=True)
        plt.title("IsolationForest Anomaly Score Distribution")
        _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_iforest_score_hist.png"))
        figs = [{
            "name":"ml_iforest_score_hist","title":"IForest Score Distribution",
            "url":f"/ml-images/{prefix}ml_iforest_score_hist.png","stage":"ml"
        }]
        logger.info(f"[ML FIG] IForest-ScoreHist -> /ml-images/{prefix}ml_iforest_score_hist.png")

        # (2) 시계열 스코어(가능하면)
        if time_col and time_col in df_in.columns:
            t = pd.to_datetime(df_in[time_col], errors="coerce")
            s = pd.Series(scores, index=t).dropna()
            if not s.empty:
                plt.figure(figsize=(10.5, 3.6))
                try:
                    s.resample("1min").mean().plot()
                except Exception:
                    s.plot()
                plt.title("IsolationForest Anomaly Score over Time")
                plt.xlabel("time"); plt.ylabel("score")
                _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_iforest_score_ts.png"))
                figs.append({
                    "name":"ml_iforest_score_ts","title":"IForest Score over Time",
                    "url":f"/ml-images/{prefix}ml_iforest_score_ts.png","stage":"ml"
                })
                logger.info(f"[ML FIG] IForest-ScoreTS -> /ml-images/{prefix}ml_iforest_score_ts.png")

        # (3) 임계치 스윕
        perc = np.linspace(0.90, 0.995, 12)
        ths = np.quantile(scores, perc)
        flagged = [(float(p), int((scores >= th).sum())) for p, th in zip(perc, ths)]
        plt.figure(figsize=(5.5, 4.0))
        plt.plot([p for p,_ in flagged], [c for _,c in flagged], marker="o")
        plt.title("Flagged Count vs Quantile Threshold")
        plt.xlabel("Quantile"); plt.ylabel("# flagged")
        _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_iforest_threshold_sweep.png"))
        figs.append({
            "name":"ml_iforest_threshold_sweep","title":"Flagged vs Threshold (Quantile)",
            "url":f"/ml-images/{prefix}ml_iforest_threshold_sweep.png","stage":"ml"
        })
        logger.info(f"[ML FIG] IForest-Threshold -> /ml-images/{prefix}ml_iforest_threshold_sweep.png")

        run_id = prefix.rstrip("_")
        summary = {
            "status": "unsupervised",
            "algo": "IsolationForest",
            "n_features": int(len(feat_cols)),
            "figs": [f["name"] for f in figs],
            "note": "No valid target; produced anomaly-score diagnostics."
        }
        with open(os.path.join(ARTIFACT_DIR, f"{run_id}_ml_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"[ML] done (unsupervised): figs={len(figs)} summary_path={os.path.join(ARTIFACT_DIR, f'{run_id}_ml_summary.json')}")
        return summary, figs

    # ---------- 타깃/태스크 확정 ----------
    # 업로드에서 공백/대소문자 이슈 정규화 (이미 했더라도 안전망)
    import re as _re
    df.columns = [_re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    if target_col:
        # 소문자 매핑으로 실제 컬럼명 교정
        col_map = {str(c).lower(): c for c in df.columns}
        key = target_col.strip().lower()
        target_col = col_map.get(key, target_col)

    # 자동 감지
    if not target_col or not task_type or target_col not in df.columns:
        logger.warning("[ML] supervised not possible → fallback to unsupervised (missing/invalid target or task)")
        return _unsupervised(df)

    # ---------- 피처/타깃 분리 + y NaN 제거 ----------
    X_all = df.drop(columns=[target_col], errors="ignore")
    feat_cols = _numeric_feature_cols(X_all)
    if len(feat_cols) == 0:
        logger.warning("[ML] no numeric features → fallback to unsupervised")
        return _unsupervised(df)

    X = df[feat_cols].copy()
    y = df[target_col].copy()

    # y 결측 제거
    na_mask = y.notna()
    dropped = int((~na_mask).sum())
    if dropped > 0:
        logger.info(f"[ML] drop rows with y NaN: {dropped}")
    X, y = X.loc[na_mask], y.loc[na_mask]
    if len(y) == 0:
        logger.warning("[ML] all rows dropped due to y NaN → fallback to unsupervised")
        return _unsupervised(df)

    # task_type 자동 교정(오입력 방지)
    if pd.api.types.is_numeric_dtype(y):
        inferred = "classification" if y.nunique(dropna=True) <= 10 else "regression"
    else:
        inferred = "classification"
    if task_type != inferred:
        logger.info(f"[ML] task_type corrected: {task_type} -> {inferred}")
        task_type = inferred

    # 분류일 때 클래스 수 검사
    if task_type == "classification" and y.nunique(dropna=True) < 2:
        logger.warning("[ML] single class after y-NaN drop → fallback to unsupervised")
        return _unsupervised(df)

    # ---------- train/test split ----------
    if isinstance(df.index, pd.DatetimeIndex):
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        split_mode = "time-ordered 80/20"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == "classification" and y.nunique(dropna=True) <= 20 else None
        )
        split_mode = "random 80/20"

    # ---------- 전처리 ----------
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test), columns=feat_cols, index=X_test.index)

    # 분류에서 문자열 라벨 인코딩
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_train):
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train), index=y_train.index)
        y_test  = pd.Series(le.transform(y_test), index=y_test.index)

    # ---------- 모델 선택 ----------
    if task_type == "classification":
        if model_type == "mlp":
            model = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                              solver="adam", max_iter=300, random_state=42)
            )
        else:
            model = RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced"
            )
    else:
        if model_type == "mlp":
            model = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                             solver="adam", max_iter=500, random_state=42)
            )
        else:
            model = RandomForestRegressor(
                n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
            )

    # ---------- 학습 ----------
    model.fit(X_train_imp, y_train)
    logger.info(f"[ML] trained model={model.__class__.__name__} task={task_type}")

    # ---------- 성능 & 시각화 ----------
    if task_type == "classification":
        # 예측/확률
        y_pred = model.predict(X_test_imp)
        proba = None
        try:
            final_est = model[-1] if hasattr(model, "__getitem__") else model
            if hasattr(final_est, "predict_proba") and y.nunique(dropna=True) == 2:
                proba = model.predict_proba(X_test_imp)[:, 1]
        except Exception as e:
            logger.warning(f"[ML] predict_proba not available: {e}")

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="binary" if y.nunique(dropna=True)==2 else "macro")),
        }
        if proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            metrics["avg_precision"] = float(average_precision_score(y_test, proba))

        # ROC/PR(이진일 때)
        if proba is not None:
            plt.figure(figsize=(5.5, 4.2))
            RocCurveDisplay.from_predictions(y_test, proba)
            plt.title("ROC Curve (Test)")
            _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_roc.png"))
            ml_figs.append({"name":"ml_roc","title":"ML ROC Curve","url":f"/ml-images/{prefix}ml_roc.png","stage":"ml"})
            logger.info(f"[ML FIG] ROC -> /ml-images/{prefix}ml_roc.png")

            plt.figure(figsize=(5.5, 4.2))
            PrecisionRecallDisplay.from_predictions(y_test, proba)
            plt.title("Precision-Recall Curve (Test)")
            _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_pr.png"))
            ml_figs.append({"name":"ml_pr","title":"ML Precision-Recall Curve","url":f"/ml-images/{prefix}ml_pr.png","stage":"ml"})
            logger.info(f"[ML FIG] PR  -> /ml-images/{prefix}ml_pr.png")

        # 혼동행렬
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4.6, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.title("Confusion Matrix (Test)")
        plt.xlabel("Predicted"); plt.ylabel("True")
        _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_confusion.png"))
        ml_figs.append({"name":"ml_confusion","title":"ML Confusion Matrix","url":f"/ml-images/{prefix}ml_confusion.png","stage":"ml"})
        logger.info(f"[ML FIG] CM  -> /ml-images/{prefix}ml_confusion.png")

    else:
        # 회귀
        y_pred = model.predict(X_test_imp)
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }

        # Pred vs True
        plt.figure(figsize=(5.5, 4.2))
        plt.scatter(y_test, y_pred, s=8)
        plt.xlabel("True"); plt.ylabel("Predicted")
        plt.title("Predicted vs True (Test)")
        _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_pred_vs_true.png"))
        ml_figs.append({"name":"ml_pred_vs_true","title":"ML Predicted vs True","url":f"/ml-images/{prefix}ml_pred_vs_true.png","stage":"ml"})
        logger.info(f"[ML FIG] PredVsTrue -> /ml-images/{prefix}ml_pred_vs_true.png")

        # Residuals hist
        residuals = y_test - y_pred
        plt.figure(figsize=(5.5, 4.2))
        sns.histplot(residuals, bins=50, kde=True)
        plt.title("Residuals Distribution (Test)")
        _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_residuals_hist.png"))
        ml_figs.append({"name":"ml_residuals_hist","title":"ML Residuals Histogram","url":f"/ml-images/{prefix}ml_residuals_hist.png","stage":"ml"})
        logger.info(f"[ML FIG] ResidHist  -> /ml-images/{prefix}ml_residuals_hist.png")

        # Residuals vs Pred
        plt.figure(figsize=(5.5, 4.2))
        plt.scatter(y_pred, residuals, s=8)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted"); plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted (Test)")
        _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_residuals_vs_pred.png"))
        ml_figs.append({"name":"ml_residuals_vs_pred","title":"ML Residuals vs Predicted","url":f"/ml-images/{prefix}ml_residuals_vs_pred.png","stage":"ml"})
        logger.info(f"[ML FIG] ResidVsPred -> /ml-images/{prefix}ml_residuals_vs_pred.png")

    # Feature Importance(가능할 때)
    top_feats = []
    final_model = model[-1] if hasattr(model, "__getitem__") else model
    if hasattr(final_model, "feature_importances_"):
        imp = final_model.feature_importances_
        order = np.argsort(imp)[::-1][:20]
        plt.figure(figsize=(6.2, 4.2))
        sns.barplot(x=imp[order], y=np.array(feat_cols)[order])
        plt.title("Feature Importance (Top 20)")
        _save_fig(os.path.join(ML_IMAGE_DIR, f"{prefix}ml_feature_importance.png"))
        ml_figs.append({"name":"ml_feature_importance","title":"ML Feature Importance (Top 20)","url":f"/ml-images/{prefix}ml_feature_importance.png","stage":"ml"})
        logger.info(f"[ML FIG] FeatImp    -> /ml-images/{prefix}ml_feature_importance.png")
        top_feats = [(feat_cols[i], float(imp[i])) for i in order]

    # 요약 저장
    ml_summary = {
        "status": "ok",
        "split_mode": split_mode,
        "task_type": task_type,
        "target_col": target_col,
        "n_features": int(len(feat_cols)),
        "n_train": int(len(X_train_imp)),
        "n_test": int(len(X_test_imp)),
        "metrics": metrics,
        "top_features": top_feats,
        "model": ("MLP" if model_type == "mlp" else "RandomForest"),
    }
    run_id = prefix.rstrip("_")
    out_path = os.path.join(ARTIFACT_DIR, f"{run_id}_ml_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ml_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"[ML] done: figs={len(ml_figs)} summary_path={out_path}")
    return ml_summary, ml_figs

# -------------------------
# API 스키마
# -------------------------
class FigureInfo(BaseModel):
    name: str
    title: str
    url: str
    stage: Optional[str] = None  # overall/stage1/.../ml

class UploadResponse(BaseModel):
    run_id: str
    n_rows: int
    n_numeric_sensors: int
    images: List[FigureInfo]  # stage 포함
    top_std_sensors: List[str]

class ReportRequest(BaseModel):
    objective: str = "EDA + ML 기반 공정 센서 진단 및 예측 성능 평가"
    language: str = "ko"  # 'ko' or 'en'

class ReportResponse(BaseModel):
    report_markdown: str

# -------------------------
# 엔드포인트: 데이터 업로드 → EDA → ML
# -------------------------
@app.post("/upload-data", response_model=UploadResponse, tags=["EDA"])
async def upload_data(
    file: UploadFile = File(...),
    time_col: Optional[str] = Form(None),
    group_cols: Optional[str] = Form(None),  # "tool,chamber" 형태
    top_n_sensors: int = Form(6),
    target_col: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),  # "classification" | "regression"
):
    if not any(file.filename.lower().endswith(ext) for ext in [".csv", ".xlsx", ".xls", ".parquet", ".txt"]):
        raise HTTPException(status_code=400, detail="CSV/XLSX/Parquet/txt 파일만 허용됩니다.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        df = _read_any(tmp_path)
        df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
        gcols = [c.strip() for c in group_cols.split(",")] if group_cols else None
        
        logger.info(f"[UPLOAD] file={file.filename} shape={df.shape}")
        logger.info(f"[UPLOAD] columns(normalized)={list(df.columns)[:50]}")  # 컬럼 최대 50개 미리보기
        logger.info(f"[UPLOAD] params: target_col={target_col} task_type={task_type} group_cols={gcols}")

        run_id = uuid.uuid4().hex[:8]
        prefix = f"{run_id}_"
        
        target_col = (target_col or "").strip() or None
        task_type  = (task_type or "").strip() or None

        # --- 1) EDA ---
        summary, figures = await run_in_threadpool(
            run_eda_pipeline,
            df=df,
            time_col=time_col,
            group_cols=gcols,
            top_n_sensors=top_n_sensors,
            prefix=prefix
        )
        logger.info("[EDA DONE]")

        # --- 2) ML (전처리/학습/평가/시각화) ---
        ml_summary, ml_figs = await run_in_threadpool(
            run_modeling_pipeline,
            df=df if time_col is None else df.reset_index(),  # 모델링은 컬럼 기반이 편하니 index 복원
            time_col=time_col,
            group_cols=gcols,
            prefix=prefix,
            target_col=target_col,
            task_type=task_type
        )
        # 요약에 ML 포함
        summary["ml"] = ml_summary
        figures.extend(ml_figs)
        logger.info(f"[ML DONE] status={ml_summary.get('status')} figs={len(ml_figs)}")

        global LAST_RUN_SUMMARY, LAST_RUN_FIGURES, LAST_RUN_ID
        LAST_RUN_SUMMARY = summary
        LAST_RUN_FIGURES = figures
        LAST_RUN_ID = run_id

        ov = summary.get("overall", {})
        # stage None 방지
        safe_figs = [{**f, "stage": (f.get("stage") or "overall")} for f in figures]

        return UploadResponse(
            run_id=run_id,
            n_rows=int(ov.get("n_rows", 0)),
            n_numeric_sensors=int(ov.get("n_numeric_sensors", 0)),
            images=safe_figs,
            top_std_sensors=ov.get("top_std_sensors", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA/ML 처리 오류: {str(e)}")

# -------------------------
# 엔드포인트: LLM 리포트 생성 (Markdown 저장 + HTML 프리뷰 파일 생성)
# -------------------------
@app.post("/generate-report", response_model=ReportResponse, tags=["Report"])
async def generate_report(req: ReportRequest, request: Request):
    if LAST_RUN_SUMMARY is None or LAST_RUN_FIGURES is None or LAST_RUN_ID is None:
        raise HTTPException(status_code=400, detail="먼저 /upload-data 로 EDA를 수행하세요.")

    # --- 요약 꺼내기 ---
    ov = LAST_RUN_SUMMARY.get("overall", {})                 # overall 요약
    stages = LAST_RUN_SUMMARY.get("stages", [])              # ["stage1", "stage2", ...]
    per_stage = LAST_RUN_SUMMARY.get("per_stage", {})        # 각 stage 요약 dict
    ml = LAST_RUN_SUMMARY.get("ml", {})                      # ML 요약

    # --- Figure manifest (stage 라벨 포함) ---
    manifest_lines = []
    for i, fig in enumerate(LAST_RUN_FIGURES, start=1):
        stage_label = fig.get("stage") or "overall"
        manifest_lines.append(f"- [Figure {i}] [{stage_label}] {fig['title']}: {fig['url']}")

    # --- Stage별 한 줄 요약 ---
    stage_summ_lines = []
    for st in stages:
        s = per_stage.get(st, {}) or {}
        top_vars = ", ".join((s.get("top_std_sensors") or [])[:5])
        oc = s.get("outlier_counts") or {}
        oc_head = dict(list(sorted(oc.items(), key=lambda kv: kv[1], reverse=True))[:5])
        stage_summ_lines.append(
            f"{st}: rows={s.get('n_rows', 0)}, sensors={s.get('n_numeric_sensors', 0)}, "
            f"top-var=[{top_vars}], outliers_top={json.dumps(oc_head, ensure_ascii=False)}"
        )
    stage_block = "\n".join(stage_summ_lines) if stage_summ_lines else "No per-stage summary."

    # --- ML 요약 블록 ---
    if ml.get("status") == "ok":
        ml_metrics_lines = [f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}" for k, v in ml.get("metrics", {}).items()]
        tf_lines = [f"{name} ({imp:.4f})" for name, imp in ml.get("top_features", [])[:10]]
        ml_block = f"""
Task Type: **{ml.get('task_type')}**
Target: **{ml.get('target_col')}**
Split: **{ml.get('split_mode')}** | Train={ml.get('n_train')} / Test={ml.get('n_test')}
Features: **{ml.get('n_features')}**

**Metrics (Test):**
{os.linesep.join(ml_metrics_lines) if ml_metrics_lines else 'N/A'}

**Top Features:**
{", ".join(tf_lines) if tf_lines else 'N/A'}
""".strip()
    else:
        ml_block = f"ML_SKIPPED: {ml.get('reason','unknown')}"

    # --- 리포트 컨텍스트 (EDA는 간결, ML 강조) ---
    context = f"""
=== OVERALL (brief) ===
Rows: {ov.get('n_rows', 0)} | Numeric Sensors: {ov.get('n_numeric_sensors', 0)}
Top-variance Sensors: {', '.join(ov.get('top_std_sensors', []))}

=== PER STAGE SUMMARY (brief) ===
{stage_block}

=== MODELING SUMMARY ===
{ml_block}

=== FIGURE MANIFEST ===
{os.linesep.join(manifest_lines)}
""".strip()

    lang = "Korean" if req.language == "ko" else "English"

    # --------------------- 프롬프트 (Markdown 강제 + ML 강조, EDA 축약) ---------------------
    prompt = f"""
You are a semiconductor process analytics report writer. Produce a professional report in {lang}.

# OUTPUT & FORMATTING CONTRACT
- Output **MUST be pure GitHub-Flavored Markdown**.
- **Do NOT** wrap the whole output in code fences (no ``` or ```markdown).
- No prose or JSON before/after the Markdown body.
- Start with a single H1 title line.
- Every embedded image must be followed by a one-line *italic* caption.
- Use ONLY figures from the manifest and context.

# VISUALIZATION REQUIREMENTS
- Be visualization-heavy with emphasis on **model evaluation** figures.
- OVERALL/EDA section should be **brief** (only a few key figures).
- For **Modeling/Evaluation**:
  - For classification: embed **ROC**, **Precision-Recall**, **Confusion Matrix**, and **Feature Importance** figures if present.
  - For regression: embed **Predicted vs True**, **Residuals Histogram**, **Residuals vs Predicted**, and **Feature Importance**.
- For each **Stage** (stage1, stage2, ...): include a short subsection with only the most relevant EDA figures (≤ 3 per stage).
- At the end, add an **Appendix – Figure Gallery** to include any important figures not used above.

# STRUCTURE
1) Executive Summary (reference 2–3 critical figures)
2) Data & Methods (EDA overview in 3–4 lines; preprocessing steps; train/test split policy)
3) **Modeling Approach & Preprocessing** (target, task, features; handling of missing data)
4) **Model Performance & Error Analysis** (embed evaluation figures; discuss metrics; reference [Figure N])
5) Key Findings by Stage (brief; ≤ 3 figures per stage)
6) Recommended Actions (operational thresholds; how to act on predictions)
7) Model Monitoring & Next Steps (drift signals, retrain cadence, key sensors/control limits)
8) Appendix – Figure Gallery (leftover important figures)

# TABLES & NUMBERS
- Add a small table summarizing test metrics.
- If available, list top features with approximate importances.

# IMAGE EMBEDDING RULE
- Always cite figures inline like [Figure N] **and** embed them using Markdown:
  ![](URL)
  *Short italic caption (1~2 sentences).*

# CONTEXT
{context}

Return **only** valid Markdown in {lang}, with no surrounding code fences.
"""

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,   # 보수적 설정 → 형식 안정성↑
        top_p=0.2,
        presence_penalty=0,
        frequency_penalty=0,
    )
    md_text = llm.invoke([HumanMessage(content=prompt)]).content

    # --- Harden: strip accidental code fences if model still adds them ---
    if md_text.strip().startswith("```"):
        md_text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", md_text.strip(), flags=re.DOTALL)

    # 이미지 URL을 절대경로로 보정해 브라우저에서 바로 보이도록
    base_url = str(request.base_url)  # e.g., http://localhost:8000/
    md_text = _prefix_image_urls(md_text, base_url)

    # 파일 저장 (md + html)
    run_id = LAST_RUN_ID
    md_path = os.path.join(ARTIFACT_DIR, f"{run_id}_report.md")
    html_path = os.path.join(ARTIFACT_DIR, f"{run_id}_report.html")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    html_body = md.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    html_full = f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>EDA+ML Report - {run_id}</title>
  <style>
    body {{ max-width: 980px; margin: 40px auto; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; line-height: 1.6; }}
    pre {{ background: #f7f7f7; padding: 12px; overflow: auto; }}
    code {{ background: #f1f1f1; padding: 2px 4px; border-radius: 4px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    h1, h2, h3 {{ margin-top: 1.4em; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #fafafa; }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
""".strip()

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_full)

    return ReportResponse(report_markdown=md_text)

# -------------------------
# 리포트 프리뷰/다운로드 엔드포인트
# -------------------------
@app.get("/report/preview", response_class=HTMLResponse, tags=["Report"])
async def report_preview():
    if LAST_RUN_ID is None:
        raise HTTPException(status_code=404, detail="리포트가 아직 생성되지 않았습니다.")
    html_path = os.path.join(ARTIFACT_DIR, f"{LAST_RUN_ID}_report.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="리포트 HTML이 없습니다. /generate-report를 먼저 실행하세요.")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/report/{run_id}", response_class=HTMLResponse, tags=["Report"])
async def report_by_id(run_id: str):
    html_path = os.path.join(ARTIFACT_DIR, f"{run_id}_report.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="해당 run_id 리포트가 없습니다.")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/report/{run_id}.md", response_class=FileResponse, tags=["Report"])
async def download_report_md(run_id: str):
    md_path = os.path.join(ARTIFACT_DIR, f"{run_id}_report.md")
    if not os.path.exists(md_path):
        raise HTTPException(status_code=404, detail="해당 run_id 마크다운이 없습니다.")
    return FileResponse(md_path, media_type="text/markdown", filename=f"{run_id}_report.md")

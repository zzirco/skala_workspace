import pandas as pd
import numpy as np

# ===== 경로 =====
PRED_PATH = "/mnt/nas4/mhj/lstm/re_saved_model/De_chamber_temperature_full_predictions.csv"     # 예측 CSV (길 수 있음)
ORIG_PATH = "/mnt/nas4/mhj/lstm/drift_prediction_results_vol2/De_chamber_temperature_drift_prediction.csv"  # 원본(100행)
OUT_PATH  = "/mnt/nas4/mhj/lstm/re_saved_model/De_chamber_temperature_matched0.csv"

# ===== 모델 입력 윈도우 =====
SEQ_LEN = 8  # 원본에서 예측 가능한 시작 오프셋

# ===== 유틸 =====
def _to_naive_ts(sr):
    ts = pd.to_datetime(sr, errors="coerce")
    try:
        return ts.dt.tz_localize(None)
    except Exception:
        return ts

def infer_tolerance(orig_ts: pd.Series) -> pd.Timedelta:
    """원본 샘플 간격의 60%를 허용오차로 사용"""
    ts = pd.to_datetime(orig_ts, errors="coerce").dropna().sort_values().unique()
    if len(ts) < 3:
        return pd.Timedelta("1s")
    diffs = np.diff(ts)
    med = np.median(diffs)
    if med <= np.timedelta64(0, "ns"):
        return pd.Timedelta("1s")
    return pd.Timedelta(med) * 0.6

# ===== 로드 & 기본 전처리 =====
pred = pd.read_csv(PRED_PATH)
orig = pd.read_csv(ORIG_PATH)

pred["timestamp"] = _to_naive_ts(pred["timestamp"])
orig["timestamp"] = _to_naive_ts(orig["timestamp"])

pred = pred.sort_values("timestamp").reset_index(drop=True)
orig = orig.sort_values("timestamp").reset_index(drop=True)

# 원본에서 예측 가능한 타임스탬프(SEQ_LEN 이후 구간)
if len(orig) <= SEQ_LEN:
    raise ValueError(f"원본 길이({len(orig)})가 SEQ_LEN({SEQ_LEN}) 이하라 예측 매칭이 불가합니다.")
orig_pred_ts = orig["timestamp"].iloc[SEQ_LEN:].dropna().reset_index(drop=True)

# 1) asof(가까운 시각 매칭) 시도
tol = infer_tolerance(orig["timestamp"])
tmp_pred = pred.rename(columns={"timestamp": "ts_pred"})
left = pd.DataFrame({"timestamp": orig_pred_ts})
right = tmp_pred[["ts_pred", "input_data", "actual", "predicted"]].dropna(subset=["ts_pred"]).sort_values("ts_pred")
merged = pd.merge_asof(
    left.sort_values("timestamp"),
    right,
    left_on="timestamp",
    right_on="ts_pred",
    direction="nearest",
    tolerance=tol
)

matched_asof = merged.dropna(subset=["ts_pred"]).drop(columns=["ts_pred"]).copy()
asof_ratio = len(matched_asof) / max(1, len(orig_pred_ts))

# 2) 매칭율이 낮으면 → 폴백: 원본 TS 덮어쓰기(앞에서부터 길이 맞춰 자름)
if asof_ratio < 0.8:
    # 사용할 길이 n = min(예측행, 원본 예측가능행)
    n = min(len(pred), len(orig_pred_ts))
    if n == 0:
        raise ValueError("예측/원본 중 하나가 비어 폴백 매칭도 불가합니다.")
    matched = pred.iloc[:n].copy()
    matched["timestamp"] = orig_pred_ts.iloc[:n].values
else:
    matched = matched_asof

# 최종 정렬 & 저장
matched = matched.sort_values("timestamp").reset_index(drop=True)
matched.to_csv(OUT_PATH, index=False)

print(f"✅ 저장 완료 → {OUT_PATH}")
print(f" - asof 매칭율: {asof_ratio*100:.1f}% (허용오차={tol})")
print(f" - 최종 행 수: {len(matched)}")
matched.head()

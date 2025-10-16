# -*- coding: utf-8 -*-
import os, json, math, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================
# 경로/하드코딩
# =============================
BASE_DIR  = "saved_model"      # ← 이전(기준) 모델 세트 위치 (.keras/.pkl/.json)
MODEL_DIR = "re_saved_model"   # ← 재학습 산출물 저장 위치

# 기본값 (base 모델이 있으면 base의 seq_len을 따름)
DEFAULT_SEQ_LEN = 8
DENSIFY_FACTOR  = 10  # 보간 배율(5~20 권장)
EPOCHS_PRETRAIN = 120
EPOCHS_FINETUNE = 40
BATCH_SIZE      = 64
LR_PRETRAIN     = 1e-3
LR_FINETUNE     = 1e-4

# 분할 비율: 75%(pretrain) / 20%(finetune) / 5%(holdout)
SPLIT_PRE = 0.75
SPLIT_FT  = 0.95

SENSORS = [
    {
        "key": "chamber_temperature",
        "csv_path": "drift_prediction_results_vol2/De_chamber_temperature_drift_prediction.csv",
        "column": "actual",
        "artifacts_prefix": "De_chamber_temperature",
    },
    {
        "key": "gas_flow_rate",
        "csv_path": "drift_prediction_results_vol2/De_gas_flow_rate_drift_prediction.csv",
        "column": "actual",
        "artifacts_prefix": "De_gas_flow_rate",
    },
    {
        "key": "rf_power",
        "csv_path": "drift_prediction_results_vol2/De_rf_power_drift_prediction.csv",
        "column": "actual",
        "artifacts_prefix": "De_rf_power",
    },
]

# =============================
# 유틸
# =============================
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0: return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100.0)

def make_supervised(series: np.ndarray, seq_len: int):
    """단일(스케일된) 시계열 → (X,y) 감독학습 세트"""
    if len(series) <= seq_len:
        return np.zeros((0, seq_len, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    n = len(X)
    X = np.array(X, dtype=np.float32).reshape(n, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(n, 1)
    return X, y

def build_lstm(seq_len: int, hidden: int = 96, dropout: float = 0.25, lr: float = 1e-3) -> keras.Model:
    """새 모델을 만들어야 할 때만 사용(베이스가 없을 때)"""
    inp = layers.Input(shape=(seq_len, 1))
    x = layers.LSTM(hidden, return_sequences=False)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden//2, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model

def detect_time_column(df: pd.DataFrame):
    for c in ["timestamp", "time", "datetime", "date"]:
        if c in df.columns:
            return c
    return None

def densify_with_interpolation(values: np.ndarray, timestamps: pd.Series | None, factor: int):
    """
    선형 보간으로 factor배 확장(학습용 구간만).
    timestamp가 있으면 median 간격 기반 리샘플 후 time-interp, 없으면 인덱스 기반 interp.
    """
    if factor <= 1 or len(values) < 2:
        return values.copy(), (pd.to_datetime(timestamps).values if timestamps is not None else None)

    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        if len(ts) < 2:
            return values.copy(), ts.values
        deltas = (ts[1:].values.astype("datetime64[ns]") - ts[:-1].values.astype("datetime64[ns]")).astype("timedelta64[ns]").astype(np.int64)
        base_ns = int(np.median(deltas)) if len(deltas) > 0 else 0
        if base_ns <= 0:
            # fallback: 인덱스 보간
            x = np.arange(len(values))
            x_new = np.linspace(0, len(values)-1, num=(len(values)-1)*factor+1)
            v_new = np.interp(x_new, x, values)
            return v_new.astype(float), None
        fine_ns = max(1, base_ns // factor)
        new_index = pd.date_range(start=ts.iloc[0], end=ts.iloc[-1], freq=pd.to_timedelta(fine_ns, unit="ns"))
        df_tmp = pd.DataFrame({"v": values}, index=ts)
        df_res = df_tmp.reindex(df_tmp.index.union(new_index)).sort_index().interpolate(method="time").reindex(new_index)
        return df_res["v"].values.astype(float), new_index.values
    else:
        x = np.arange(len(values))
        x_new = np.linspace(0, len(values)-1, num=(len(values)-1)*factor+1)
        v_new = np.interp(x_new, x, values)
        return v_new.astype(float), None

def split_indices(n: int, pre_ratio: float, ft_ratio: float):
    """총 길이 n → [0:pre) pretrain / [pre:ft) finetune / [ft:n) holdout"""
    pre_idx = max(1, int(n * pre_ratio))
    ft_idx  = max(pre_idx + 1, int(n * ft_ratio))
    ft_idx  = min(ft_idx, n - 1)  # holdout 최소 1포인트 보장
    return pre_idx, ft_idx

def base_artifacts(prefix: str):
    """saved_model 경로의 베이스 아티팩트 경로 사전"""
    return {
        "model":  os.path.join(BASE_DIR,  f"{prefix}_model.keras"),
        "scaler": os.path.join(BASE_DIR,  f"{prefix}_scaler.pkl"),
        "config": os.path.join(BASE_DIR,  f"{prefix}_config.json"),
    }

def load_base_if_exists(prefix: str):
    """베이스(.keras/.pkl/.json)를 로드. 없으면(None, None, None) 반환."""
    paths = base_artifacts(prefix)
    if not all(os.path.exists(p) for p in paths.values()):
        return None, None, None
    base_model = tf.keras.models.load_model(paths["model"])
    with open(paths["scaler"], "rb") as f:
        base_scaler = pickle.load(f)
    with open(paths["config"], "r", encoding="utf-8") as f:
        base_config = json.load(f)
    return base_model, base_scaler, base_config

# =============================
# 재학습 파이프라인 (한 센서, 베이스 warm-start)
# =============================
def retrain_one_sensor(csv_path: str, value_col: str, artifacts_prefix: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "="*90)
    print(f"[{artifacts_prefix}] 재학습 시작 (warm-start from saved_model if available)")
    print("="*90)

    # 0) 베이스 로드 시도
    base_model, base_scaler, base_config = load_base_if_exists(artifacts_prefix)
    if base_model is not None:
        # 베이스 모델 입력 길이 감지
        input_shape = base_model.inputs[0].shape  # (None, seq_len, 1)
        base_seq_len = int(input_shape[1])
        seq_len = base_seq_len
        print(f"[{artifacts_prefix}] ✅ Base model found. Using base seq_len={seq_len}")
    else:
        seq_len = DEFAULT_SEQ_LEN
        print(f"[{artifacts_prefix}] ⚠️ Base model not found. Train from scratch with seq_len={seq_len}")

    # 1) 데이터 로드/정렬
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        raise ValueError(f"[{artifacts_prefix}] '{value_col}' 컬럼이 없습니다. CSV 컬럼: {list(df.columns)}")
    time_col = detect_time_column(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

    series_raw = df[value_col].astype(float).values
    n_all = len(series_raw)
    print(f"[{artifacts_prefix}] 원시 포인트: {n_all}, 범위: {np.min(series_raw):.4f} ~ {np.max(series_raw):.4f}")
    if n_all < seq_len + 10:
        print(f"[{artifacts_prefix}] ⚠️ 데이터가 매우 짧습니다. (n={n_all}) — 보간 확대 사용 권장")

    # 2) 75/95 분할
    pre_idx, ft_idx = split_indices(n_all, SPLIT_PRE, SPLIT_FT)
    pre_raw = series_raw[:pre_idx]
    ft_raw  = series_raw[pre_idx:ft_idx]
    val_raw = series_raw[ft_idx:]

    ts_series = df[time_col] if time_col is not None else None
    pre_ts = ts_series.iloc[:pre_idx] if ts_series is not None else None
    ft_ts  = ts_series.iloc[pre_idx:ft_idx] if ts_series is not None else None
    val_ts = ts_series.iloc[ft_idx:] if ts_series is not None else None

    print(f"[{artifacts_prefix}] 분할 → pre:{len(pre_raw)}  finetune:{len(ft_raw)}  val(holdout):{len(val_raw)}")

    # 3) 보간(학습용만)
    pre_dense, pre_dense_ts = densify_with_interpolation(pre_raw, pre_ts, DENSIFY_FACTOR)
    ft_dense,  ft_dense_ts  = densify_with_interpolation(ft_raw,  ft_ts,  DENSIFY_FACTOR)
    print(f"[{artifacts_prefix}] 보간 → pre:{len(pre_dense)}  finetune:{len(ft_dense)}  (val 원시:{len(val_raw)})")

    # 4) 스케일러: 새로 fit (권장) — 드리프트 적응을 위해 학습 집합 기준
    #    (정말 필요하면 base_scaler를 그대로 써도 되지만, 분포가 달라졌다면 성능 저하 위험)
    scaler = MinMaxScaler()
    train_all = np.concatenate([pre_dense, ft_dense]) if len(ft_dense) > 0 else pre_dense
    scaler.fit(train_all.reshape(-1, 1))

    pre_sc = scaler.transform(pre_dense.reshape(-1, 1)).reshape(-1)
    ft_sc  = scaler.transform(ft_dense.reshape(-1, 1)).reshape(-1)
    val_sc = scaler.transform(val_raw.reshape(-1, 1)).reshape(-1)

    # 5) 감독학습 데이터
    X_pre, y_pre = make_supervised(pre_sc, seq_len)
    X_ft,  y_ft  = make_supervised(ft_sc,  seq_len)

    print(f"[{artifacts_prefix}] supervised → X_pre:{X_pre.shape}  X_ft:{X_ft.shape}")

    # 6) 모델: base가 있으면 그대로 사용(가중치 포함), 없으면 새로 빌드
    if base_model is not None:
        model = base_model
        # 학습률 설정 후 compile
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(LR_PRETRAIN)), loss="mse")
        print(f"[{artifacts_prefix}] ▶ Warm-start pretrain ...")
    else:
        model = build_lstm(seq_len=seq_len, hidden=96, dropout=0.25, lr=LR_PRETRAIN)
        print(f"[{artifacts_prefix}] ▶ Scratch pretrain ...")

    # Pretrain (끝까지)
    cbs_pre = [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=1e-5)]
    model.fit(
        X_pre, y_pre,
        epochs=EPOCHS_PRETRAIN,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=2,
        callbacks=cbs_pre,
    )

    # Finetune (최근 20% 적응; 학습률 낮춰 계속)
    if X_ft.shape[0] > 0:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(LR_FINETUNE)), loss="mse")
        cbs_ft = [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-5)]
        print(f"[{artifacts_prefix}] ▶ Finetune ...")
        model.fit(
            X_ft, y_ft,
            epochs=EPOCHS_FINETUNE,
            batch_size=BATCH_SIZE,
            shuffle=False,
            verbose=2,
            callbacks=cbs_ft,
        )
    else:
        print(f"[{artifacts_prefix}] ⚠️ Finetune 구간이 비어 건너뜀")

    # 7) 전체 구간 1-step 예측 생성 (CSV 저장용)
    # full_values = pre_dense + ft_dense + val_raw (ft_dense가 없으면 pre_dense + val_raw)
    full_values = np.concatenate([pre_dense, ft_dense, val_raw]) if len(ft_dense) > 0 else np.concatenate([pre_dense, val_raw])
    full_scaled = scaler.transform(full_values.reshape(-1, 1)).reshape(-1)
    X_full, y_full = make_supervised(full_scaled, seq_len)
    if X_full.shape[0] == 0:
        raise ValueError(f"[{artifacts_prefix}] 전체 구간이 너무 짧아 예측 생성 불가")
    y_full_hat = model.predict(X_full, verbose=0).reshape(-1)

    # 역변환
    y_full_inv     = scaler.inverse_transform(y_full.reshape(-1,1)).flatten()
    y_full_hat_inv = scaler.inverse_transform(y_full_hat.reshape(-1,1)).flatten()

    # timestamp & input_data 정렬
    def _mk_ts(ts):
        if ts is None: return None
        return pd.to_datetime(ts)

    pre_dense_ts = _mk_ts(pre_dense_ts)
    ft_dense_ts  = _mk_ts(ft_dense_ts)
    val_ts       = _mk_ts(val_ts)

    if pre_dense_ts is None and ft_dense_ts is None and val_ts is None:
        full_ts = np.arange(len(full_values))
    else:
        parts = []
        # pre
        parts.append(pd.Series(pre_dense_ts) if pre_dense_ts is not None else pd.Series(np.arange(len(pre_dense))))
        # ft
        if len(ft_dense) > 0:
            parts.append(pd.Series(ft_dense_ts) if ft_dense_ts is not None else pd.Series(np.arange(len(ft_dense))))
        # val
        if len(val_raw) > 0:
            parts.append(pd.Series(val_ts) if val_ts is not None else pd.Series(np.arange(len(val_raw))))
        full_ts = pd.concat(parts, ignore_index=True).values

    ts_for_pred = full_ts[seq_len:]
    input_data  = full_values[seq_len-1 : -1]
    actual      = y_full_inv
    predicted   = y_full_hat_inv

    # 8) 저장(모델/스케일러/컨피그/CSV) → re_saved_model/
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path  = os.path.join(MODEL_DIR, f"{artifacts_prefix}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{artifacts_prefix}_scaler.pkl")
    config_path = os.path.join(MODEL_DIR, f"{artifacts_prefix}_config.json")
    csv_path    = os.path.join(MODEL_DIR, f"{artifacts_prefix}_full_predictions.csv")

    # 새 모델/스케일러 저장
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # config 저장 (base 여부/seq_len 기록)
    cfg = {
        "base_used": base_model is not None,
        "seq_len": int(seq_len),
        "densify_factor": int(DENSIFY_FACTOR),
        "splits": {"pre": SPLIT_PRE, "finetune": SPLIT_FT, "val": 1.0 - SPLIT_FT},
        "columns": {"target": value_col},
        "source_csv": csv_path.replace("_full_predictions.csv", "_SOURCE.csv"),  # 의미 표시용 텍스트
        "notes": "full_predictions.csv = 전체 구간 1-step 예측; timestamp,input_data,actual,predicted",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 예측 CSV 저장 (timestamp,input_data,actual,predicted)
    out_df = pd.DataFrame({
        "timestamp": ts_for_pred,
        "input_data": input_data,
        "actual": actual,
        "predicted": predicted
    })
    if np.issubdtype(out_df["timestamp"].dtype, np.datetime64):
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"])
    out_df.to_csv(csv_path, index=False)

    print(f"[{artifacts_prefix}] 저장 완료 →")
    print(f"  Base used : {base_model is not None}")
    print(f"  Model     : {model_path}")
    print(f"  Scaler    : {scaler_path}")
    print(f"  Config    : {config_path}")
    print(f"  CSV       : {csv_path}")

    # 간단 지표(전체 구간)
    r = {
        "sensor": artifacts_prefix,
        "used_base": bool(base_model is not None),
        "seq_len": int(seq_len),
        "rows_csv": len(out_df),
        "csv": csv_path
    }
    print(f"[{artifacts_prefix}] summary: {r}")
    return r

# =============================
# 메인
# =============================
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 90)
    print("🌡 공정 센서 재학습 (BASE 모델 warm-start) → 전체 예측 CSV 저장")
    print("=" * 90)

    summary = []
    for s in SENSORS:
        res = retrain_one_sensor(
            csv_path=s["csv_path"],
            value_col=s["column"],          # 'actual'
            artifacts_prefix=s["artifacts_prefix"]
        )
        summary.append(res)

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(MODEL_DIR, "retrain_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("✅ 모든 센서 재학습 완료 (warm-start) & 전체 예측 CSV 저장")
    print("=" * 90)
    print(f"📊 요약 CSV: {summary_path}")
    print(summary_df)

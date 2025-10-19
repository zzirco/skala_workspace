# -*- coding: utf-8 -*-
import os, json, math, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# GPU 설정
print("="*80)
print("GPU 확인 중...")
print("="*80)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 발견: {len(gpus)}개")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"⚠️  GPU 설정 에러: {e}")
else:
    print("❌ GPU를 찾을 수 없습니다. CPU로 실행됩니다.")
print("="*80)

# =============================
# 경로/하드코딩
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

DEFAULT_SEQ_LEN = 8
EPOCHS_TRAIN = 5
BATCH_SIZE = 64
LR = 1e-3

# 재학습 구간: 2001~4000 (인덱스 2000~3999)
RETRAIN_START_IDX = 2000  # 2001번째 데이터 (인덱스 2000)

SENSORS = [
    {
        "key": "chamber_temperature",
        "csv_path": DATA_DIR / "De_chamber_temperature.csv",
        "column": "actual",
        "artifacts_prefix": "De_chamber_temperature",
    },
    {
        "key": "gas_flow_rate",
        "csv_path": DATA_DIR / "De_gas_flow_rate.csv",
        "column": "actual",
        "artifacts_prefix": "De_gas_flow_rate",
    },
    {
        "key": "rf_power",
        "csv_path": DATA_DIR / "De_rf_power.csv",
        "column": "actual",
        "artifacts_prefix": "De_rf_power",
    },
]

# =============================
# 유틸
# =============================
def make_supervised(series: np.ndarray, seq_len: int):
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

def base_artifacts(prefix: str):
    return {
        "model":  ARTIFACTS_DIR / "model" / f"{prefix}_model.keras",
        "scaler": ARTIFACTS_DIR / "scaler" / f"{prefix}_scaler.pkl",
        "config": ARTIFACTS_DIR / "config" / f"{prefix}_config.json",
    }


def load_base_if_exists(prefix: str):
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
# 재학습 파이프라인
# =============================
def retrain_one_sensor(csv_path: str, value_col: str, artifacts_prefix: str):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("\n" + "="*90)
    print(f"[{artifacts_prefix}] 재학습 시작")
    print("="*90)

    # 0) 베이스 로드
    base_model, base_scaler, base_config = load_base_if_exists(artifacts_prefix)
    if base_model is not None:
        input_shape = base_model.inputs[0].shape
        base_seq_len = int(input_shape[1])
        seq_len = base_seq_len
        print(f"[{artifacts_prefix}] ✅ Base model found. Using base seq_len={seq_len}")
    else:
        seq_len = DEFAULT_SEQ_LEN
        print(f"[{artifacts_prefix}] ⚠️ Base model not found. Train from scratch with seq_len={seq_len}")

    # 1) 데이터 로드
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        raise ValueError(f"[{artifacts_prefix}] '{value_col}' 컬럼이 없습니다.")
    
    time_col = detect_time_column(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

    series_raw = df[value_col].astype(float).values
    n_all = len(series_raw)
    print(f"[{artifacts_prefix}] 전체 포인트: {n_all}")

    # 2) 구간 분할: 1~2000 (인덱스 0~1999), 2001~4000 (인덱스 2000~3999)
    original_data = series_raw[:RETRAIN_START_IDX]  # 1~2000
    retrain_data = series_raw[RETRAIN_START_IDX:]    # 2001~4000

    print(f"[{artifacts_prefix}] 원본 구간(1~2000): {len(original_data)}개")
    print(f"[{artifacts_prefix}] 재학습 구간(2001~{n_all}): {len(retrain_data)}개")

    # 3) 재학습 구간으로만 스케일러 fit 및 학습
    scaler = MinMaxScaler()
    scaler.fit(retrain_data.reshape(-1, 1))
    retrain_sc = scaler.transform(retrain_data.reshape(-1, 1)).reshape(-1)

    X_train, y_train = make_supervised(retrain_sc, seq_len)
    print(f"[{artifacts_prefix}] supervised → X_train:{X_train.shape}")

    # 4) 모델 재학습
    if base_model is not None:
        model = base_model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(LR)), loss="mse")
        print(f"[{artifacts_prefix}] ▶ Warm-start 재학습...")
    else:
        model = build_lstm(seq_len=seq_len, hidden=96, dropout=0.25, lr=LR)
        print(f"[{artifacts_prefix}] ▶ Scratch 학습...")

    cbs = [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=1e-5)]
    model.fit(
        X_train, y_train,
        epochs=EPOCHS_TRAIN,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=2,
        callbacks=cbs,
    )

    # 5) 재학습 구간 예측
    y_train_hat = model.predict(X_train, verbose=0).reshape(-1)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
    y_train_hat_inv = scaler.inverse_transform(y_train_hat.reshape(-1,1)).flatten()

    # 6) CSV 생성: 1~2000(원본) + 2001~4000(예측)
    # 원본 구간
    if time_col is not None:
        ts_original = df[time_col].iloc[:RETRAIN_START_IDX].values
    else:
        ts_original = np.arange(len(original_data))
    
    df_original = pd.DataFrame({
        "timestamp": ts_original,
        "actual": original_data,
        "predicted": original_data,  # 원본 그대로
        "phase": "original"
    })

    # 재학습 구간 (seq_len만큼 건너뛰어야 함)
    if time_col is not None:
        ts_retrain = df[time_col].iloc[RETRAIN_START_IDX + seq_len:].values
    else:
        ts_retrain = np.arange(RETRAIN_START_IDX + seq_len, n_all)

    df_retrain = pd.DataFrame({
        "timestamp": ts_retrain,
        "actual": y_train_inv,
        "predicted": y_train_hat_inv,
        "phase": "retrained"
    })

    # 합치기
    result_df = pd.concat([df_original, df_retrain], ignore_index=True)

    # 7) 저장
    # CSV는 artifacts/predicton 폴더에 저장
    csv_dir = os.path.join(ARTIFACTS_DIR, "predictions")
    os.makedirs(csv_dir, exist_ok=True)

    # 모델 vol 번호와 동일하게 CSV도 버전 관리
    vol_num = 1
    while True:
        model_path = os.path.join(ARTIFACTS_DIR, "model", f"{artifacts_prefix}_model_vol{vol_num}.keras")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler", f"{artifacts_prefix}_scaler_vol{vol_num}.pkl")
        config_path = os.path.join(ARTIFACTS_DIR, "config", f"{artifacts_prefix}_config_vol{vol_num}.json")
        csv_path = os.path.join(csv_dir, f"{artifacts_prefix}_prediction_vol{vol_num}.csv")
        
        if not os.path.exists(model_path):
            break
        vol_num += 1
    
    # 모델은 vol2, vol3... 형식으로 저장
    vol_num = 2
    while True:
        model_path = os.path.join(ARTIFACTS_DIR, "model", f"{artifacts_prefix}_model_vol{vol_num}.keras")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler", f"{artifacts_prefix}_scaler_vol{vol_num}.pkl")
        config_path = os.path.join(ARTIFACTS_DIR, "config", f"{artifacts_prefix}_config_vol{vol_num}.json")
        
        if not os.path.exists(model_path):
            break
        vol_num += 1

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    cfg = {
        "base_used": base_model is not None,
        "seq_len": int(seq_len),
        "retrain_start_idx": RETRAIN_START_IDX,
        "columns": {"target": value_col},
        "version": f"vol{vol_num}"
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if time_col is not None:
        result_df["timestamp"] = pd.to_datetime(result_df["timestamp"])
    result_df.to_csv(csv_path, index=False)

    print(f"[{artifacts_prefix}] 저장 완료")
    print(f"  Model: {model_path}")
    print(f"  CSV: {csv_path}")
    print(f"  원본(1~2000): {len(df_original)}개")
    print(f"  재학습 예측(2001~): {len(df_retrain)}개")

    return {
        "sensor": artifacts_prefix,
        "used_base": bool(base_model is not None),
        "seq_len": int(seq_len),
        "rows_csv": len(result_df),
        "csv": csv_path,
        "version": f"vol{vol_num}"
    }

# =============================
# 메인
# =============================
if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("=" * 90)
    print("🌡 공정 센서 재학습 (2001~4000 구간)")
    print("=" * 90)

    summary = []
    for s in SENSORS:
        res = retrain_one_sensor(
            csv_path=s["csv_path"],
            value_col=s["column"],
            artifacts_prefix=s["artifacts_prefix"]
        )
        summary.append(res)

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(ARTIFACTS_DIR, "retrain_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("✅ 모든 센서 재학습 완료")
    print("=" * 90)
    print(summary_df)
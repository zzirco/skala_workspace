"""
새로운 LSTM 온도 예측 모델용 모듈
- 입력 데이터: interpolated_temperature_1000.csv
- 모델 파일: MODEL_SAVE_PATH_NEW (config.py에 정의)
- 출력: 예측 결과(JSON 또는 그래프)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import math
import os
from datetime import datetime
from config import IMAGE_DIR, MODEL_SAVE_PATH_NEW, DATA_PATH_NEW, MODEL_SHAPES_PLOT_PATH, PREDICTION_PLOT_PATH

# -------------------------------------------------
# RMSE 계산
# -------------------------------------------------
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    result_msg = f"The root mean squared error is {rmse:.4f}."
    print(result_msg)
    return result_msg

# -------------------------------------------------
# 예측 결과 그래프 시각화
# -------------------------------------------------
def plot_predictions(timestamps, real, predicted):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, real, color='red', label='Real Temperature')
    plt.plot(timestamps, predicted, color='blue', label='Predicted Temperature')
    plt.title('Temperature Forecasting (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    save_path = os.path.join(IMAGE_DIR, f"temperature_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(save_path)
    return save_path

# -------------------------------------------------
# 모델 예측 + 결과 JSON 반환
# -------------------------------------------------
def process_to_json(dataset: pd.DataFrame):
    """
    Timestamp, temperature 2컬럼 데이터로 LSTM 예측 수행
    """
    model = load_model(MODEL_SAVE_PATH_NEW)

    timestamps = dataset["Timestamp"].tolist()
    values = dataset["temperature"].values.reshape(-1, 1)

    sc = MinMaxScaler(feature_range=(0, 1))
    scaled = sc.fit_transform(values)

    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    predicted = model.predict(X)
    predicted = sc.inverse_transform(predicted)
    y_true = sc.inverse_transform(y.reshape(-1, 1))

    rmse_msg = return_rmse(y_true, predicted)
    plot_path = plot_predictions(timestamps[lookback:], y_true.flatten(), predicted.flatten())

    return {
        "timestamps": timestamps[lookback:],
        "real": y_true.flatten().tolist(),
        "predicted": predicted.flatten().tolist(),
        "rmse": rmse_msg,
        "plot_path": plot_path
    }


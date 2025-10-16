import numpy as np
import pandas as pd
import pickle
import json
import tensorflow as tf
from datetime import datetime, timedelta

class TemperaturePredictionModel:
    """온도 예측 모델 래퍼 클래스"""
    
    def __init__(self, model_dir='model'):
        """
        모델 초기화
        
        Parameters:
        -----------
        model_dir : str
            모델과 관련 파일들이 저장된 디렉토리
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.config = None
        
        self._load_model()
        self._load_scaler()
        self._load_config()
    
    def _load_model(self):
        """Keras 모델 로드"""
        model_path = f"{self.model_dir}/temperature_prediction_model.keras"
        print(f"📥 모델 로딩 중: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ 모델 로드 완료")
    
    def _load_scaler(self):
        """MinMaxScaler 로드"""
        scaler_path = f"{self.model_dir}/scaler.pkl"
        print(f"📥 스케일러 로딩 중: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✅ 스케일러 로드 완료")
    
    def _load_config(self):
        """모델 설정 로드"""
        config_path = f"{self.model_dir}/model_config.json"
        print(f"📥 설정 로딩 중: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        print("✅ 설정 로드 완료")
        print(f"   - SEQ_LEN: {self.config['seq_len']}")
        print(f"   - AUGMENT_FACTOR: {self.config['augment_factor']}")
    
    def preprocess_input(self, data):
        """
        입력 데이터 전처리
        
        Parameters:
        -----------
        data : array-like
            원본 온도 데이터 (1D array)
            
        Returns:
        --------
        preprocessed : array
            전처리된 데이터 (shape: [1, SEQ_LEN, 1])
        """
        seq_len = self.config['seq_len']
        
        # 최근 SEQ_LEN개만 사용
        if len(data) < seq_len:
            raise ValueError(f"데이터가 부족합니다. 최소 {seq_len}개 필요, 현재 {len(data)}개")
        
        recent_data = data[-seq_len:]
        
        # 스케일링
        scaled_data = self.scaler.transform(recent_data.reshape(-1, 1))
        
        # 모델 입력 형태로 변환: [batch_size=1, seq_len, features=1]
        preprocessed = scaled_data.reshape(1, seq_len, 1)
        
        return preprocessed
    
    def predict_next(self, data):
        """
        다음 1스텝 예측
        
        Parameters:
        -----------
        data : array-like
            입력 온도 데이터
            
        Returns:
        --------
        prediction : float
            예측된 다음 온도값
        """
        # 전처리
        X = self.preprocess_input(data)
        
        # 예측
        pred_scaled = self.model.predict(X, verbose=0)
        
        # 역변환
        prediction = self.scaler.inverse_transform(pred_scaled)[0, 0]
        
        return prediction
    
    def predict_multi_step(self, data, n_steps=10):
        """
        여러 스텝 예측 (재귀적 예측)
        
        Parameters:
        -----------
        data : array-like
            입력 온도 데이터
        n_steps : int
            예측할 스텝 수
            
        Returns:
        --------
        predictions : list
            예측된 온도값 리스트
        """
        seq_len = self.config['seq_len']
        
        # 초기 시퀀스 준비
        current_sequence = np.array(data[-seq_len:], dtype=np.float32)
        predictions = []
        
        print(f"\n🔮 {n_steps}스텝 예측 시작...")
        
        for step in range(n_steps):
            # 1스텝 예측
            next_pred = self.predict_next(current_sequence)
            predictions.append(next_pred)
            
            # 시퀀스 업데이트 (가장 오래된 값 제거, 새 예측값 추가)
            current_sequence = np.append(current_sequence[1:], next_pred)
            
            if (step + 1) % 10 == 0:
                print(f"   Step {step+1}/{n_steps} 완료")
        
        print("✅ 예측 완료")
        return predictions
    
    def predict_from_csv(self, csv_path, n_steps=10, output_path=None, 
                        time_interval_min=15, time_interval_max=20):
        """
        CSV 파일에서 데이터를 읽어 예측
        
        Parameters:
        -----------
        csv_path : str
            입력 CSV 파일 경로
        n_steps : int
            예측할 스텝 수
        output_path : str, optional
            결과를 저장할 CSV 경로
        time_interval_min : int
            최소 시간 간격 (초)
        time_interval_max : int
            최대 시간 간격 (초)
            
        Returns:
        --------
        result_df : DataFrame
            예측 결과 데이터프레임
        """
        print(f"\n📂 CSV 로딩: {csv_path}")
        
        # CSV 읽기
        df = pd.read_csv(csv_path)
        
        # 컬럼명 확인
        temp_col = None
        possible_cols = ['Chamber_Temperature', 'temperature', 'temp', 'Temperature']
        for col in possible_cols:
            if col in df.columns:
                temp_col = col
                break
        
        if temp_col is None:
            temp_col = df.columns[0]  # 첫 번째 컬럼 사용
            print(f"⚠️  온도 컬럼을 찾을 수 없어 첫 번째 컬럼 사용: {temp_col}")
        else:
            print(f"✅ 온도 컬럼 발견: {temp_col}")
        
        data = df[temp_col].values
        print(f"   데이터 포인트: {len(data)}개")
        print(f"   데이터 범위: {data.min():.2f}° ~ {data.max():.2f}°")
        
        # 예측 수행
        predictions = self.predict_multi_step(data, n_steps=n_steps)
        
        # 타임스탬프 생성
        future_timestamps = []
        if 'timestamp' in df.columns or 'time' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
            last_time = pd.to_datetime(df[time_col].iloc[-1])
            
            print(f"✅ 타임스탬프 컬럼 발견: {time_col}")
            print(f"   마지막 시간: {last_time}")
            
            # 랜덤 간격으로 미래 타임스탬프 생성 (15~20초)
            current_time = last_time
            for i in range(n_steps):
                interval = np.random.randint(time_interval_min, time_interval_max + 1)
                current_time += timedelta(seconds=interval)
                future_timestamps.append(current_time)
            
            print(f"   예측 시작: {future_timestamps[0]}")
            print(f"   예측 종료: {future_timestamps[-1]}")
            
            # 결과 데이터프레임 생성 (타임스탬프 포함)
            result_df = pd.DataFrame({
                'timestamp': future_timestamps,
                'predicted_temperature': predictions
            })
        else:
            print(f"⚠️  타임스탬프 컬럼이 없습니다. step 번호만 사용합니다.")
            # 결과 데이터프레임 생성 (타임스탬프 없음)
            result_df = pd.DataFrame({
                'step': range(1, n_steps + 1),
                'predicted_temperature': predictions
            })
        
        # 결과 저장
        if output_path:
            result_df.to_csv(output_path, index=False)
            print(f"\n💾 결과 저장: {output_path}")
        
        return result_df


# ============================================================================
# 사용 예제
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Temperature Prediction System")
    print("="*70)
    
    # 1. 모델 초기화
    predictor = TemperaturePredictionModel(model_dir='model')
    
    print("\n" + "="*70)
    print("예측 시작")
    print("="*70)
    
    # 2. CSV에서 예측
    # 방법 1: 기본 사용
    result = predictor.predict_from_csv(
        csv_path='interpolated_temperature_1000.csv',
        n_steps=20,  # 20스텝 예측
        output_path='prediction_results.csv'
    )
    
    # 결과 출력
    print("\n📊 예측 결과:")
    print(result.head(10))
    print("...")
    print(result.tail(5))
    
    print("\n📈 예측 통계:")
    print(f"   평균: {result['predicted_temperature'].mean():.2f}°")
    print(f"   최소: {result['predicted_temperature'].min():.2f}°")
    print(f"   최대: {result['predicted_temperature'].max():.2f}°")
    print(f"   표준편차: {result['predicted_temperature'].std():.2f}°")
    
    # 3. 직접 배열로 예측 (옵션)
    print("\n" + "="*70)
    print("배열 직접 입력 예제")
    print("="*70)
    
    # 임의의 데이터 생성 (실제로는 센서 데이터)
    sample_data = np.random.uniform(70, 80, 1000)
    
    # 단일 스텝 예측
    next_temp = predictor.predict_next(sample_data)
    print(f"다음 온도 예측: {next_temp:.2f}°")
    
    # 멀티 스텝 예측
    future_temps = predictor.predict_multi_step(sample_data, n_steps=5)
    print(f"향후 5스텝 예측: {[f'{t:.2f}°' for t in future_temps]}")
    
    print("\n" + "="*70)
    print("완료!")
    print("="*70)


# ============================================================================
# 백엔드 API 예제 (FastAPI)
# ============================================================================
"""
# main.py (FastAPI 백엔드)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io

app = FastAPI()

# 모델 전역 로드
predictor = TemperaturePredictionModel(model_dir='saved_model')

@app.post("/predict")
async def predict_temperature(
    file: UploadFile = File(...),
    n_steps: int = 10
):
    '''
    온도 예측 API
    
    Parameters:
    - file: CSV 파일 (온도 데이터)
    - n_steps: 예측할 스텝 수
    '''
    try:
        # CSV 읽기
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # 온도 데이터 추출
        temp_col = 'Chamber_Temperature'  # 또는 자동 감지
        data = df[temp_col].values
        
        # 예측
        predictions = predictor.predict_multi_step(data, n_steps=n_steps)
        
        # 결과 반환
        return JSONResponse({
            "status": "success",
            "n_steps": n_steps,
            "predictions": [float(p) for p in predictions],
            "statistics": {
                "mean": float(np.mean(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions))
            }
        })
    
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.get("/model-info")
async def get_model_info():
    '''모델 정보 조회'''
    return JSONResponse({
        "seq_len": predictor.config['seq_len'],
        "augment_factor": predictor.config['augment_factor'],
        "performance": predictor.config['performance']
    })
"""
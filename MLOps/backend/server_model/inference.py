import numpy as np
import pandas as pd
import pickle
import json
import tensorflow as tf
from pathlib import Path

class TemperaturePredictionModel:
    """온도 예측 모델 래퍼 클래스"""
    
    def __init__(self, model_path, scaler_path, config_path):
        """
        모델 초기화
        
        Parameters:
        -----------
        model_path : str
            모델 파일 경로 (.keras)
        scaler_path : str
            스케일러 파일 경로 (.pkl)
        config_path : str
            설정 파일 경로 (.json)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self.model = None
        self.scaler = None
        self.config = None
        
        self._load_model()
        self._load_scaler()
        self._load_config()
    
    def _load_model(self):
        """Keras 모델 로드"""
        print(f"📥 모델 로딩 중: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("✅ 모델 로드 완료")
    
    def _load_scaler(self):
        """MinMaxScaler 로드"""
        print(f"📥 스케일러 로딩 중: {self.scaler_path}")
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✅ 스케일러 로드 완료")
    
    def _load_config(self):
        """모델 설정 로드"""
        print(f"📥 설정 로딩 중: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        print("✅ 설정 로드 완료")
        print(f"   - SEQ_LEN: {self.config['seq_len']}")
    
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
    
    def predict_from_csv(self, csv_path, output_path=None):
        """
        CSV 파일에서 데이터를 읽어 예측 (기존 타임스탬프 유지)
        
        Parameters:
        -----------
        csv_path : str
            입력 CSV 파일 경로 (timestamp와 temperature 포함)
        output_path : str, optional
            결과를 저장할 CSV 경로
            
        Returns:
        --------
        result_df : DataFrame
            예측 결과가 추가된 데이터프레임
        """
        print(f"\n📂 CSV 로딩: {csv_path}")
        
        # CSV 읽기
        df = pd.read_csv(csv_path)
        
        print(f"✅ 데이터 로드 완료")
        print(f"   총 행 수: {len(df)}")
        print(f"   컬럼: {list(df.columns)}")
        
        # 컬럼명 확인
        temp_col = None
        possible_cols = ['Chamber_Temperature', 'chamber_temperature', 'temperature', 'temp', 'Temperature']
        for col in possible_cols:
            if col in df.columns:
                temp_col = col
                break
        
        if temp_col is None:
            # timestamp가 아닌 첫 번째 컬럼 사용
            for col in df.columns:
                if col.lower() not in ['timestamp', 'time', 'date']:
                    temp_col = col
                    break
            print(f"⚠️  온도 컬럼을 찾을 수 없어 '{temp_col}' 컬럼 사용")
        else:
            print(f"✅ 온도 컬럼 발견: {temp_col}")
        
        # 실제 온도 데이터
        actual_temps = df[temp_col].values
        print(f"   데이터 범위: {actual_temps.min():.2f} ~ {actual_temps.max():.2f}")
        
        # 배치 예측 수행 (속도 최적화)
        seq_len = self.config['seq_len']
        predictions = np.full(len(df), np.nan)
        
        print(f"\n🔮 배치 예측 시작...")
        print(f"   (처음 {seq_len}개는 실제값 사용)")
        
        # 예측 가능한 모든 시퀀스를 한 번에 준비
        valid_indices = range(seq_len, len(df))
        n_valid = len(valid_indices)
        
        if n_valid > 0:
            # 배치 데이터 준비
            X_batch = np.zeros((n_valid, seq_len, 1))
            for batch_idx, i in enumerate(valid_indices):
                X_batch[batch_idx] = actual_temps[i-seq_len:i].reshape(seq_len, 1)
            
            # 스케일링
            X_batch_scaled = self.scaler.transform(X_batch.reshape(-1, 1)).reshape(n_valid, seq_len, 1)
            
            # 한 번에 배치 예측
            print(f"   배치 예측 수행 중... ({n_valid}개)")
            pred_scaled = self.model.predict(X_batch_scaled, batch_size=256, verbose=1)
            
            # 역스케일링
            pred_values = self.scaler.inverse_transform(pred_scaled).flatten()
            
            # 결과 저장
            predictions[seq_len:] = pred_values
        
        print("✅ 예측 완료")
        
        # 결과 데이터프레임 생성 (timestamp, actual, predicted, error)
        result_df = pd.DataFrame()
        
        # timestamp 컬럼 찾기
        timestamp_col = None
        for col in ['timestamp', 'time', 'Timestamp', 'Time', 'date', 'Date']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            result_df['timestamp'] = df[timestamp_col]
        else:
            # timestamp가 없으면 인덱스 사용
            result_df['timestamp'] = range(len(df))
            print(f"⚠️  타임스탬프 컬럼을 찾을 수 없어 인덱스 사용")
        
        result_df['actual'] = actual_temps
        result_df['predicted'] = predictions
        
        # NaN 값(처음 seq_len개)을 실제값으로 채우기
        print(f"\n🔧 처음 {seq_len}개 행의 predicted를 actual 값으로 채우는 중...")
        result_df.loc[result_df['predicted'].isna(), 'predicted'] = result_df.loc[result_df['predicted'].isna(), 'actual']
        print(f"✅ 채우기 완료 (인덱스 0~{seq_len-1})")
        
        # 예측 오차 계산 (actual - predicted)
        result_df['error'] = result_df['actual'] - result_df['predicted']
        
        # 통계 출력 (실제 예측값에 대해서만 - seq_len 이후)
        if n_valid > 0:
            valid_predictions = result_df.iloc[seq_len:]
            valid_errors = valid_predictions['error']
            print(f"\n📊 예측 통계 (실제 예측 데이터 {n_valid}개, 인덱스 {seq_len}~{len(df)-1}):")
            print(f"   MAE (평균 절대 오차): {np.abs(valid_errors).mean():.6f}")
            print(f"   RMSE (평균 제곱근 오차): {np.sqrt((valid_errors**2).mean()):.6f}")
            print(f"   최대 오차: {np.abs(valid_errors).max():.6f}")
            print(f"   최소 오차: {np.abs(valid_errors).min():.6f}")
        
        # 결과 저장
        if output_path:
            # 디렉토리가 없으면 생성
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            print(f"\n💾 결과 저장: {output_path}")
        
        return result_df


# ============================================================================
# 사용 예제 - Le 챔버 3개 파라미터 모두 예측
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Li Chamber - Multi Parameter Prediction System")
    print("="*70)
    
    # 기본 경로 설정
    base_dir = Path('artifacts')
    model_dir = base_dir / 'model'
    config_dir = base_dir / 'config'
    scaler_dir = base_dir / 'scaler'
    predictions_dir = base_dir / 'predictions'
    
    # Le 챔버의 3개 파라미터 정의
    parameters = [
        {
            'name': 'chamber_temperature',
            'model': model_dir / 'Li_chamber_temperature_model.keras',
            'scaler': scaler_dir / 'Li_chamber_temperature_scaler.pkl',
            'config': config_dir / 'Li_chamber_temperature_config.json',
            'input_csv': 'data/Li_chamber_temperature.csv',
            'output_csv': predictions_dir / 'Li_chamber_temperature_prediction.csv'
        },
        {
            'name': 'gas_flow_rate',
            'model': model_dir / 'Li_gas_flow_rate_model.keras',
            'scaler': scaler_dir / 'Li_gas_flow_rate_scaler.pkl',
            'config': config_dir / 'Li_gas_flow_rate_config.json',
            'input_csv': 'data/Li_gas_flow_rate.csv',
            'output_csv': predictions_dir / 'Li_gas_flow_rate_prediction.csv'
        },
        {
            'name': 'rf_power',
            'model': model_dir / 'Li_rf_power_model.keras',
            'scaler': scaler_dir / 'Li_rf_power_scaler.pkl',
            'config': config_dir / 'Li_rf_power_config.json',
            'input_csv': 'data/Li_rf_power.csv',
            'output_csv': predictions_dir / 'Li_rf_power_prediction.csv'
        }
    ]
    # 각 파라미터에 대해 예측 수행
    results_summary = []
    
    for i, param in enumerate(parameters, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/3] {param['name'].upper()} 예측 시작")
        print(f"{'='*70}")
        
        try:
            # 모델 초기화
            predictor = TemperaturePredictionModel(
                model_path=str(param['model']),
                scaler_path=str(param['scaler']),
                config_path=str(param['config'])
            )
            
            # 예측 수행
            result = predictor.predict_from_csv(
                csv_path=param['input_csv'],
                output_path=str(param['output_csv'])
            )
            
            # 결과 미리보기
            print(f"\n📊 {param['name']} 예측 결과 미리보기:")
            print("\n처음 10개 행 (실제값으로 채워짐):")
            print(result.head(10))
            print("\n예측이 시작되는 부분 (인덱스 180~190):")
            print(result.iloc[180:191])
            
            results_summary.append({
                'parameter': param['name'],
                'status': '✅ 성공',
                'total_rows': len(result),
                'predicted_rows': len(result) - predictor.config['seq_len']
            })
            
            print(f"\n✅ {param['name']} 예측 완료!")
            
        except Exception as e:
            print(f"\n❌ {param['name']} 예측 실패: {str(e)}")
            results_summary.append({
                'parameter': param['name'],
                'status': f'❌ 실패: {str(e)}',
                'total_rows': 0,
                'predicted_rows': 0
            })
            import traceback
            traceback.print_exc()
    
    # 최종 요약
    print("\n" + "="*70)
    print("전체 예측 완료 - 요약")
    print("="*70)
    
    for summary in results_summary:
        print(f"\n📌 {summary['parameter']}:")
        print(f"   상태: {summary['status']}")
        if summary['total_rows'] > 0:
            print(f"   전체 행: {summary['total_rows']}")
            print(f"   실제 예측 행: {summary['predicted_rows']}")
    
    print(f"\n📁 결과 저장 위치: {predictions_dir}")
    print("   - Li_chamber_temperature_prediction.csv")
    print("   - Li_gas_flow_rate_prediction.csv")
    print("   - Li_rf_power_prediction.csv")
    
    print("\n" + "="*70)
    print("완료!")
    print("="*70)
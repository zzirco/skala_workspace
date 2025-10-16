#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.
#### 제공되는 실습 코드는 완성된 버전이 아니며, 일부 이스터 에그 (개선이 필요한 발견 사항)을 포함하고 있습니다.

# pip install fastapi "uvicorn[standard]" pandas pytz python-multipart
# pip install -U pip wheel
# pip install matplotlib

'''설치 패키지 설명 :
# fastapi, uvicorn[standard] : FastAPI를 통한 모델 서빙에 필요, uvicorn[standard]는 의존성 패키지까지 추가 설치
# pandas: pd (데이터프레임 처리)
# pytz: 시간대(timezone) 처리
# python-multipart: 파일 업로드 처리'''

# main.py
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from contextlib import asynccontextmanager
import asyncio
import base64
from datetime import datetime
import importlib
import os
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import pytz
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, JSONResponse
from . import model
from . import weight_used_model
# 상대 경로 사용, 현재 폴더인 servrer_model 상위 폴더에서 현 위치 인식
from fastapi.staticfiles import StaticFiles
# 정적 마운트 아래에 추가
from fastapi import Response

# uvicorn 실행 위치에 따라서, 파일 경로 식별이 달라지는 점 확인하기 (현재 디렉토리 위치는 model_serving이고, 하위에 server_model 디렉토리내에 main.py가 있다고 할 때)
# python -m uvicorn server_model.main:app --port 8001 --reload

# from . import config
# 이 경우는 상대 경로로써, 현재 실행 중인 main.py와 같은 디렉토리 위치에서 config.py 찾아서 가져오므로, 해당 파일 확인 필요
# model_serving/server_model/config.py

from config import UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR
# 이 경우는 현재 uvicorn 실행한 경로 위치인 model_serving과 같은 디렉토리 위치에서 config.py 찾아서 가져오므로, 해당 파일 확인 필요
# model_serving/config.py

from .semiconductor_agent import SemiconductorRAGAgent

# -------------------------------------------------
# 경로/디렉터리 및 프리픽스(root_path)
# -------------------------------------------------
STD_DIR = Path(__file__).resolve().parent.parent  # .../model_serving
PUBLIC_DIR = STD_DIR / "public"

# 프록시 하위 경로에서 서비스할 경우 설정 (예: /api/v2)
APP_ROOT_PATH = os.getenv("APP_ROOT_PATH", "").rstrip("/")  # 빈 문자열 또는 "/api/v2"

# 데이터 폴더 경로
DATA_DIR = Path(__file__).resolve().parent / "data"

# 타임존
timezone = pytz.timezone("Asia/Seoul")

router = APIRouter()

load_dotenv()

# -------------------------------------------------
# Lifespan: 스타트업을 가볍게 (블로킹 작업 금지)
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 정적/결과 디렉터리 보장
    for d in (PUBLIC_DIR, UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    yield
    # 종료 시 별도 정리 없음

app = FastAPI(
    lifespan=lifespan,
    root_path=APP_ROOT_PATH,           # ✅ 프리픽스 반영
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static 경로에 정적 리소스 제공
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    ico = PUBLIC_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    png = PUBLIC_DIR / "favicon.png"
    if png.exists():
        return FileResponse(str(png), media_type="image/png")
    return Response(status_code=204)  # 404 대신 조용히 처리

# 간단한 요청 로그 (디버그용)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp: Response
    try:
        resp = await call_next(request)
    finally:
        # 필요한 경우 상세 로깅 추가
        pass
    return resp

# -------------------------------------------------
# 유틸
# -------------------------------------------------
def _b64_png(path: Path) -> str:
    """PNG 파일을 data URI(base64)로 변환"""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return "data:image/png;base64," + encoded
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {e}")

async def _read_csv_async(file_path: Path) -> pd.DataFrame:
    """CSV를 스레드에서 읽기 (이벤트 루프 비블로킹)"""
    def _read():
        return pd.read_csv(file_path, index_col="Date", parse_dates=["Date"]).fillna("NaN")
    return await asyncio.to_thread(_read)

# main.py 내 또는 new_temperature_model.py 상단에 추가
async def _read_sensor_csv_async(file_path: Path) -> pd.DataFrame:
    """센서 예측용 CSV (Timestamp, Value 2컬럼) 비동기 로드"""
    def _read():
        df = pd.read_csv(file_path)
        # ✅ 컬럼 이름 확인
        expected_cols = ["timestamp", "Chamber_Temperature"]
        if len(df.columns) != 2:
            raise ValueError(f"Invalid CSV format: expected 2 columns, got {len(df.columns)}")
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"CSV must contain columns: {expected_cols}")

        # ✅ Timestamp 파싱
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        # ✅ NaN 처리
        df = df.fillna(method="ffill").fillna(method="bfill")

        return df

    return await asyncio.to_thread(_read)


# -------------------------------------------------
# 헬스체크 / 루트
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "root_path": APP_ROOT_PATH or "/"}

@app.get("/")
def root():
    """
    index.html을 반환하되, 프리픽스가 있을 경우 <base href=".../">를 주입해
    정적 자원 경로 문제를 완화합니다.
    """
    index_html = PUBLIC_DIR / "index.html"
    if not index_html.exists():
        return {"message": "public/index.html not found. Place your frontend under /public or use /static."}

    html = index_html.read_text(encoding="utf-8", errors="ignore")

    rp = APP_ROOT_PATH or "/"
    # 이미 base가 없다면 <head> 바로 뒤에 주입
    if "<base" not in html.lower():
        html = html.replace("<head>", f'<head><base href="{rp if rp.endswith("/") else rp + "/"}">', 1)

    return HTMLResponse(content=html)

# -------------------------------------------------
# 업로드/예측
# -------------------------------------------------
@router.post("/upload")
async def post_data_set(file: UploadFile = File(...)):
    """
    CSV 업로드 → 두 LSTM 모델(weight_used_model, model)로 예측 수행
    - 무거운 연산은 모두 스레드로 오프로드하여 서버 반응성 유지
    - 모델 모듈은 요청 시 동적 임포트(스타트업 블로킹 방지)
    """
    try:
        # 1) 저장 경로 구성
        current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_location = Path(UPLOAD_DIR) / new_filename

        # 2) 업로드 파일 저장
        contents = await file.read()
        await asyncio.to_thread(file_location.write_bytes, contents)

        # 3) CSV 로드
        dataset = await _read_csv_async(file_location)

        # 4) 모듈 지연 임포트
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        model_mod = importlib.import_module(".model", package=__package__)

        # 5) 예측 실행 (스레드 오프로드)
        result_visualizing_LSTM, result_evaluating_LSTM = await asyncio.to_thread(weight_mod.process, dataset)
        result_visualizing_LSTM_v2, result_evaluating_LSTM_v2 = await asyncio.to_thread(model_mod.process, dataset)

        # 6) 결과 이미지 존재 확인
        img1 = Path(result_visualizing_LSTM)
        img2 = Path(result_visualizing_LSTM_v2)
        if not img1.exists():
            raise HTTPException(status_code=500, detail=f"File not found: {img1}")
        if not img2.exists():
            raise HTTPException(status_code=500, detail=f"File not found: {img2}")

        return {
            "result_visualizing_LSTM": _b64_png(img1),
            "result_evaluating_LSTM": result_evaluating_LSTM,
            "result_visualizing_LSTM_v2": _b64_png(img2),
            "result_evaluating_LSTM_v2": result_evaluating_LSTM_v2,
            "saved_filename": new_filename,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# 다운로드/뷰
# -------------------------------------------------
@router.get("/download")
async def download():
    """weight_used_model이 생성한 stock 예측 이미지를 다운로드"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(IMAGE_DIR) / weight_mod.get_stock_png()
        if not img_name.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {img_name}")
        return FileResponse(path=str(img_name), media_type="application/octet-stream", filename="stock.png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_shapes")
async def download_model_architecture_shapes():
    """weight_used_model이 생성한 모델 구조(shapes) 이미지를 다운로드"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(IMAGE_DIR) / weight_mod.get_model_shapes_png()
        if not img_name.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {img_name}")
        return FileResponse(path=str(img_name), media_type="application/octet-stream", filename="model_shapes.png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view-download")
async def view_downloaded_image():
    """weight_used_model이 생성한 stock 예측 이미지를 HTML로 보기"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(IMAGE_DIR) / weight_mod.get_stock_png()
        img_base64 = _b64_png(img_name)
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>Downloaded Stock Prediction Image</h1>
                    <img src="{img_base64}" alt="Stock Prediction Image" />
                </body>
            </html>
            """
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    """
    CSV 파일을 업로드받아 LSTM 예측 결과를 JSON 형태로 반환합니다.
    (그래프 이미지를 생성하지 않고, Vue 프론트에서 직접 그래프를 그릴 때 사용)
    """
    try:
        # 파일 저장
        current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_location = Path(UPLOAD_DIR) / new_filename

        contents = await file.read()
        await asyncio.to_thread(file_location.write_bytes, contents)

        # CSV 읽기
        dataset = await _read_csv_async(file_location)

        # 모델 호출
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        result_json = await asyncio.to_thread(weight_mod.process_to_json, dataset)

        return result_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/sensor-data")
async def get_sensor_data(tool: str):
    """
    특정 장비(tool)의 센서 데이터를 JSON으로 반환
    예: /api/sensor-data?tool=Deposition
    """
    try:
        # tool 이름 → 파일명 매핑
        file_map = {
            "Deposition": "Deposition_data.csv",
            "Etching": "Etching_data.csv",
            "Lithography": "Lithography_data.csv",
        }

        if tool not in file_map:
            raise HTTPException(status_code=400, detail=f"Invalid tool name: {tool}")

        file_path = DATA_DIR / file_map[tool]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # CSV 읽기
        df = pd.read_csv(file_path)

        # Timestamp 컬럼이 있다면 정렬
        if "Timestamp" in df.columns:
            df = df.sort_values("Timestamp")

        # 필요한 컬럼만 선택 (존재하는 경우만)
        keep_cols = [col for col in ["Timestamp", "Chamber_Temperature", "Gas_Flow_Rate", "Vacuum_Pressure"] if col in df.columns]
        df = df[keep_cols]

        # JSON 변환
        data = df.to_dict(orient="records")
        return {"tool": tool, "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/predict-temperature-json")
async def predict_temperature_json(file: UploadFile = File(...)):
    """
    새로운 온도 예측 LSTM 모델을 사용하여 예측 결과(JSON)를 반환합니다.
    """
    try:
        current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_location = Path(UPLOAD_DIR) / new_filename

        contents = await file.read()
        await asyncio.to_thread(file_location.write_bytes, contents)

        dataset = await _read_sensor_csv_async(file_location)

        # 새 모델 모듈 로드
        new_model_mod = importlib.import_module(".temperature_model", package=__package__)
        result_json = await asyncio.to_thread(new_model_mod.process_to_json, dataset)

        return result_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/compare-sensor")
async def compare_sensor(sensor: str = "temperature", mode: str = "pred_half"):
    """
    센서 타입별 예측 vs 실제 데이터 비교 API
    mode: pred_half | real_half | real_full
    예: /compare-sensor?sensor=temperature&mode=real_half
    """
    try:
        # ✅ 센서명 → 파일명 매핑 (기존 코드 그대로)
        file_map = {
            "temperature": "De_chamber_temperature_drift_prediction.csv",
            "gas": "De_gas_flow_rate_drift_prediction.csv",
            "pressure": "De_rf_power_drift_prediction.csv",
        }

        if sensor not in file_map:
            raise HTTPException(status_code=400, detail=f"Invalid sensor name: {sensor}")

        data_path = Path(
            rf"C:\skala_workspace\MLOps\model_serving_win\server_model\data\{file_map[sensor]}"
        )

        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

        # ✅ CSV 로드 및 컬럼 검증
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        if not {"Timestamp", "predicted", "real"}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="Missing required columns (Timestamp, predicted, real)")

        total_len = len(df)
        half_len = total_len // 2

        timestamps = df["Timestamp"].tolist()
        predicted = df["predicted"].tolist()
        real = df["real"].tolist()

        # ✅ mode별 동작 처리
        if mode == "pred_half":
            # 예측 절반만
            timestamps = timestamps[:half_len]
            predicted = predicted[:half_len]
            real = [None] * half_len

        elif mode == "real_half":
            # 전체 예측 + 실제 절반 (나머지 None)
            real_half = real[:half_len]
            padding = [None] * (total_len - half_len)
            real = real_half + padding

        elif mode == "real_full":
            # 전체 예측 + 전체 실제
            pass  # 이미 full 리스트로 존재

        else:
            raise HTTPException(status_code=400, detail="Invalid mode parameter")

        # ✅ 응답 반환
        return {
            "sensor": sensor,
            "timestamps": timestamps,
            "predicted": predicted,
            "real": real,
            "info": {
                "total_points": total_len,
                "half_points": half_len,
                "mode": mode,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compare-sensor-updated")
async def compare_sensor_updated(sensor: str = "temperature"):
    """
    모델 재학습 후 예측값(업데이트 CSV)을 기반으로 전체 그래프 데이터를 반환
    - 기존 데이터는 유지하고, 이 API 호출 시에만 updated CSV를 사용
    """
    try:
        # ✅ 원본 및 업데이트 파일 경로
        base_path = Path(
            rf"C:\skala_workspace\MLOps\model_serving_win\server_model\data\De_chamber_temperature_drift_prediction.csv"
        )
        updated_path = Path(
            rf"C:\skala_workspace\MLOps\model_serving_win\server_model\data\De_chamber_temperature_drift_prediction_updated.csv"
        )

        if not base_path.exists():
            raise HTTPException(status_code=404, detail=f"Base data not found: {base_path}")
        if not updated_path.exists():
            raise HTTPException(status_code=404, detail=f"Updated data not found: {updated_path}")

        df_base = pd.read_csv(base_path)
        df_new = pd.read_csv(updated_path)

        df_base.columns = df_base.columns.str.strip()
        df_new.columns = df_new.columns.str.strip()

        if not {"Timestamp", "predicted", "real"}.issubset(df_base.columns):
            raise HTTPException(status_code=400, detail="Base file missing required columns.")
        if "predicted" not in df_new.columns:
            raise HTTPException(status_code=400, detail="Updated file missing 'predicted' column.")

        total_len = len(df_base)
        half_len = total_len // 2

        # ✅ 기존 데이터 복사 후 후반부 예측값만 업데이트
        df_updated = df_base.copy()
        new_pred_values = df_new["predicted"].values[: total_len - half_len]
        df_updated.loc[half_len:, "predicted"] = new_pred_values

        # ✅ 그래프 데이터 생성
        timestamps = df_updated["Timestamp"].tolist()
        predicted = df_updated["predicted"].tolist()
        real = df_updated["real"].tolist() if "real" in df_updated.columns else [None] * total_len

        return {
            "sensor": sensor,
            "timestamps": timestamps,
            "predicted": predicted,
            "real": real,
            "info": {
                "total_points": total_len,
                "replaced_points": len(new_pred_values),
                "message": "Updated predictions applied to back half of data"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate-report")
async def generate_report():
    """
    LLM 기반 반도체 예측 분석 리포트 생성 API
    """
    try:
        agent = SemiconductorRAGAgent(knowledge_pdf_paths=["1.pdf", "2.pdf","3.pdf","4.pdf","5.pdf","6.pdf","7.pdf"], force_reembed=False)
        agent.run_full_analysis(
            data_file_path="semiconductor_quality_control.csv",
            output_pdf_path="반도체_분석_보고서4.pdf"
        )
        return FileResponse("반도체_분석_보고서4.pdf", media_type="application/pdf", filename="analysis_report.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/update-retrained-prediction")
async def update_retrained_prediction(sensor: str = "temperature"):
    """
    재학습된 예측 CSV 기반으로 기존 예측값을 업데이트하는 API.
    예: /update-retrained-prediction?sensor=temperature
    """
    try:
        # ✅ 기존 데이터 (compare-sensor용)
        original_path = Path(
            r"C:\skala_workspace\MLOps\model_serving_win\server_model\data\De_chamber_temperature_matched.csv"
        )
        retrained_path = Path(
            r"C:\skala_workspace\MLOps\model_serving_win\server_model\data\De_chamber_temperature_drift_prediction.csv"
        )

        if not original_path.exists() or not retrained_path.exists():
            raise HTTPException(status_code=404, detail="Required data file not found.")

        df_orig = pd.read_csv(original_path)
        df_new = pd.read_csv(retrained_path)

        # ✅ 컬럼 정리
        df_orig.columns = df_orig.columns.str.strip()
        df_new.columns = df_new.columns.str.strip()

        if not {"Timestamp", "predicted"}.issubset(df_orig.columns) or not {"predicted"}.issubset(df_new.columns):
            raise HTTPException(status_code=400, detail="Missing required columns in either file.")

        total_len = len(df_orig)
        half_len = total_len // 2

        # ✅ 후반 절반을 재학습 예측값으로 교체
        df_updated = df_orig.copy()
        df_updated.loc[half_len:, "predicted"] = df_new["predicted"].values[: total_len - half_len]

        # ✅ 새로운 CSV 저장
        updated_path = (
            Path(r"C:\skala_workspace\MLOps\model_serving_win\server_model\data\De_chamber_temperature_updated.csv")
        )
        df_updated.to_csv(updated_path, index=False)

        # ✅ 응답 반환
        return {
            "status": "ok",
            "message": "Predicted values updated using retrained model output.",
            "updated_file": str(updated_path),
            "updated_rows": len(df_updated),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

# 실행 명령어 예시: 순서대로 백엔드 띄운 후, 프론트엔드 띄우기, 현재 디렉토리 server_model 상위에서 실행 (상대 경로 . 사용)
# python -m uvicorn server_model.main:app --port 8001 --reload
# http://localhost:8001/static/index.html

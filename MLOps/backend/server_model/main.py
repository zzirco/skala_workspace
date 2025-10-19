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
BASE_DIR = Path(__file__).resolve().parent          # backend/server_model
SERVER_DIR = BASE_DIR.parent / "server"             # backend/server
DATA_DIR = BASE_DIR.parent / "artifacts/predictions"                        # backend/server_model/data
PUBLIC_DIR = BASE_DIR / "public"

UPLOAD_DIR = SERVER_DIR / "uploaded_files"
IMAGE_DIR = SERVER_DIR / "view-model-architecture"
MODEL_IMG_DIR = SERVER_DIR / "model-images"

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
    root_path="/",           # ✅ 프리픽스 반영
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
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

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
            "temperature": "De_chamber_temperature_prediction_vol1.csv",
            "gas": "De_gas_flow_rate_prediction_vol1.csv",
            "pressure": "De_rf_power_prediction_vol1.csv",
        }

        if sensor not in file_map:
            raise HTTPException(status_code=400, detail=f"Invalid sensor name: {sensor}")

        data_path = DATA_DIR / file_map[sensor]

        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

        # ✅ CSV 로드 및 컬럼 검증
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        if not {"timestamp", "predicted", "actual"}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="Missing required columns (timestamp, predicted, actual)")

        total_len = len(df)
        half_len = total_len // 2

        timestamps = df["timestamp"].tolist()
        predicted = df["predicted"].tolist()
        real = df["actual"].tolist()

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
        file_map = {
            "temperature": "De_chamber_temperature_prediction_vol1.csv",
            "gas": "De_gas_flow_rate_prediction_vol1.csv",
            "pressure": "De_rf_power_prediction_vol1.csv",
        }

        if sensor not in file_map:
            raise HTTPException(status_code=400, detail=f"Invalid sensor name: {sensor}")

        data_path = DATA_DIR / file_map[sensor]
        # ✅ 원본 및 업데이트 파일 경로
        base_path = data_path
        updated_path = data_path

        if not base_path.exists():
            raise HTTPException(status_code=404, detail=f"Base data not found: {base_path}")
        if not updated_path.exists():
            raise HTTPException(status_code=404, detail=f"Updated data not found: {updated_path}")

        df_base = pd.read_csv(base_path)
        df_new = pd.read_csv(updated_path)

        df_base.columns = df_base.columns.str.strip()
        df_new.columns = df_new.columns.str.strip()

        if not {"timestamp", "predicted", "actual"}.issubset(df_base.columns):
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
        timestamps = df_updated["timestamp"].tolist()
        predicted = df_updated["predicted"].tolist()
        real = df_updated["actual"].tolist() if "actual" in df_updated.columns else [None] * total_len

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
    
@router.get("/update-retrained-prediction")
async def update_retrained_prediction(sensor: str = "temperature"):
    """
    재학습된 예측 CSV 기반으로 기존 예측값을 업데이트하는 API.
    예: /update-retrained-prediction?sensor=temperature
    """
    try:
        # ✅ 기존 데이터 (compare-sensor용)
        original_path = DATA_DIR / "De_chamber_temperature_matched.csv"
        retrained_path = DATA_DIR / "De_chamber_temperature_drift_prediction.csv"

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
        updated_path = DATA_DIR / "De_chamber_temperature_updated.csv"
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
    
def _format_compare_payload(df: pd.DataFrame, sensor: str, mode: str = "real_full"):
    """
    /compare-sensor 형식으로 DataFrame을 직렬화해 반환.
    df에는 최소 'timestamp', 'predicted' 컬럼이 있어야 하며,
    'actual'이 없으면 real은 None 리스트로 채움.
    mode: pred_half | real_half | real_full
    """
    # 컬럼 정규화
    cols = {c.strip().lower(): c for c in df.columns}
    required_pred = "predicted"
    ts_key = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else None)
    if ts_key is None:
        raise HTTPException(status_code=400, detail="CSV must contain a timestamp-like column (timestamp or date).")

    if required_pred not in cols:
        raise HTTPException(status_code=400, detail="CSV must contain 'predicted' column.")

    has_actual = "actual" in cols

    # 정렬(가능 시)
    try:
        # Timestamp를 파싱 가능한 경우 정렬
        t = pd.to_datetime(df[cols[ts_key]], errors="coerce")
        df = df.assign(_ts=t).sort_values("_ts").drop(columns=["_ts"])
    except Exception:
        pass

    total_len = len(df)
    half_len = total_len // 2

    timestamps = df[cols[ts_key]].astype(str).tolist()
    predicted = df[cols[required_pred]].tolist()
    if has_actual:
        real = df[cols["actual"]].tolist()
    else:
        real = [None] * total_len

    # mode 적용
    if mode == "pred_half":
        timestamps = timestamps[:half_len]
        predicted = predicted[:half_len]
        real = [None] * half_len
    elif mode == "real_half":
        real_half = real[:half_len]
        padding = [None] * (total_len - half_len)
        real = real_half + padding
    elif mode == "real_full":
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid mode parameter")

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

@router.get("/predict-original-model")
async def predict_original_model(sensor: str = "temperature", mode: str = "real_full"):
    """
    기존(재학습 이전) 모델로 예측을 수행하고, 결과 CSV를 저장한 뒤
    /compare-sensor와 동일 형식(sensor/timestamps/predicted/real/info)으로 응답합니다.
    mode: pred_half | real_half | real_full (기본 real_full)
    """
    try:
        import importlib
        inference_mod = importlib.import_module(".inference", package=__package__)

        model_map = {
            "temperature": {
                "model": "De_chamber_temperature_model.keras",
                "scaler": "De_chamber_temperature_scaler.pkl",
                "config": "De_chamber_temperature_config.json",
                "input_csv": "data/De_chamber_temperature.csv",
                "output_csv": "artifacts/predictions/De_chamber_temperature_prediction_vol1.csv",
            },
            "gas": {
                "model": "De_gas_flow_rate_model.keras",
                "scaler": "De_gas_flow_rate_scaler.pkl",
                "config": "De_gas_flow_rate_config.json",
                "input_csv": "data/De_gas_flow_rate.csv",
                "output_csv": "artifacts/predictions/De_gas_flow_rate_prediction_vol1.csv",
            },
            "pressure": {
                "model": "De_rf_power_model.keras",
                "scaler": "De_rf_power_scaler.pkl",
                "config": "De_rf_power_config.json",
                "input_csv": "data/De_rf_power.csv",
                "output_csv": "artifacts/predictions/De_rf_power_prediction_vol1.csv",
            },
        }

        if sensor not in model_map:
            raise HTTPException(status_code=400, detail=f"Invalid sensor name: {sensor}")

        base_dir = Path(__file__).resolve().parent.parent
        artifacts_dir = base_dir / "artifacts"
        data_dir = base_dir / "server_model" / "data"

        m = model_map[sensor]
        model_path = artifacts_dir / "model" / m["model"]
        scaler_path = artifacts_dir / "scaler" / m["scaler"]
        config_path = artifacts_dir / "config" / m["config"]
        input_csv = Path(m["input_csv"])
        output_csv = Path(m["output_csv"])

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        if not scaler_path.exists():
            raise HTTPException(status_code=404, detail=f"Scaler file not found: {scaler_path}")
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
        if not input_csv.exists():
            raise HTTPException(status_code=404, detail=f"Input CSV not found: {input_csv}")

        # 예측 수행 (동기 → 스레드)
        predictor = inference_mod.TemperaturePredictionModel(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            config_path=str(config_path)
        )
        result_df = await asyncio.to_thread(
            predictor.predict_from_csv,
            str(input_csv),
            str(output_csv)
        )

        # 결과 CSV 보장
        if not output_csv.exists():
            raise HTTPException(status_code=500, detail=f"Output CSV not created: {output_csv}")

        # 방금 생성한 CSV를 로드하여 /compare-sensor 포맷으로 직렬화
        df_out = pd.read_csv(output_csv)
        # 컬럼 정규화
        df_out.columns = df_out.columns.str.strip()

        # 일부 파이프라인은 컬럼명이 'Timestamp'로 출력될 수 있으니 보완
        # 또한 'actual'이 없을 수 있음(그 경우 real은 None으로 채움)
        payload = _format_compare_payload(df_out, sensor=sensor, mode=mode)

        # 필요하면 상태/메시지 등 메타도 함께 반환하고 싶다면 아래처럼 병합 가능
        payload["status"] = "ok"
        payload["message"] = f"Original model inference complete for '{sensor}'"
        payload["csv_path"] = str(output_csv)

        return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.encoders import jsonable_encoder

MAX_JSON_ROWS_PER_SENSOR = 0  # 0 또는 None이면 전체 행 전송

def _read_csv_as_json(path: str, max_rows: int | None = MAX_JSON_ROWS_PER_SENSOR):
    import pandas as pd, numpy as np
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {"error": f"CSV not found: {path}", "rows": 0, "records": []}
    df = pd.read_csv(p)
    # Timestamp → 문자열 정규화(있으면)
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S").fillna(df["Timestamp"].astype(str))
    if max_rows and max_rows > 0:
        df = df.head(max_rows)
    df = df.replace({np.nan: None})
    return {"rows": len(df), "records": df.to_dict(orient="records")}
    
@router.post("/retrain")
async def retrain_models():
    try:
        import importlib, asyncio, pandas as pd
        retrain_mod = importlib.import_module(".retrain", package=__package__)
        sensors = retrain_mod.SENSORS
        results = []
        for s in sensors:
            res = await asyncio.to_thread(
                retrain_mod.retrain_one_sensor,
                s["csv_path"], s["column"], s["artifacts_prefix"]
            )
            results.append({
                "sensor": s["key"],
                "version": res.get("version"),
                "csv": str(res.get("csv")),
                "rows_csv": res.get("rows_csv", None),
            })

        versions = [r["version"] for r in results if r.get("version")]
        last_version = max([int(v.replace("vol","")) for v in versions], default=1)

        summary_name = f"retrain_summary_vol{last_version}.csv"
        summary_path = retrain_mod.ARTIFACTS_DIR / summary_name
        pd.DataFrame(results).to_csv(summary_path, index=False)

        # ✅ 센서별 CSV → JSON 붙이기
        data_by_sensor = {}
        for r in results:
            data_by_sensor[r["sensor"]] = _read_csv_as_json(r["csv"])

        payload = {
            "status": "ok",
            "message": f"All sensors retrained (version {last_version}) successfully.",
            "results": results,
            "summary_csv": str(summary_path),
            "data": data_by_sensor,  # ← 프론트가 한 번에 가져감
        }
        return jsonable_encoder(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate-report")
async def generate_report():
    """
    LLM 기반 반도체 예측 분석 리포트 생성 API
    """
    try:
        data_dir = Path(r"D:\skala_workspace\MLOps\backend\data")
        data = sorted(str(p) for p in data_dir.glob("*.pdf"))
        print("[DEBUG] Knowledge PDFs:", data)
        agent = SemiconductorRAGAgent(knowledge_pdf_paths=data, force_reembed=True)
        agent.run_full_analysis(
            data_file_path="semiconductor_quality_control.csv",
            output_pdf_path="반도체_분석_보고서4.pdf"
        )
        resp = FileResponse(
            "반도체_분석_보고서4.pdf",
            media_type="application/pdf",
            filename="analysis_report.pdf",
        )
        # 👇 캐시 방지
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from typing import List

@router.post("/upload-and-reembed")
async def kb_upload_and_reembed(files: List[UploadFile] = File(...)):
    """
    여러 개 PDF를 한 번에 업로드하고, update 폴더에 저장한 뒤
    add_new_documents()로 벡터DB에 즉시 추가(재임베딩)합니다.
    """
    try:
        data_dir   = Path(r"D:\skala_workspace\MLOps\backend\data")
        update_dir = Path(r"D:\skala_workspace\MLOps\backend\update")
        update_dir.mkdir(parents=True, exist_ok=True)

        if not files:
            raise HTTPException(status_code=400, detail="업로드할 파일이 없습니다.")

        saved_paths: list[str] = []
        errors: list[dict] = []

        # 1) 업로드 저장 (여러 개)
        for f in files:
            try:
                ext = Path(f.filename).suffix.lower()
                if ext != ".pdf":
                    raise ValueError(f"PDF만 업로드 가능합니다: {f.filename}")

                ts_name = datetime.now(timezone).strftime("%Y%m%d_%H%M%S_") + f.filename
                dest = update_dir / ts_name

                content = await f.read()
                await asyncio.to_thread(dest.write_bytes, content)

                saved_paths.append(str(dest))
            except Exception as e:
                errors.append({"file": f.filename, "error": str(e)})

        if not saved_paths and errors:
            # 모두 실패한 경우
            raise HTTPException(status_code=400, detail={"message":"모든 파일 업로드 실패", "errors": errors})

        # 2) 에이전트 생성(기존 KB는 유지, 강제 재임베딩 X)
        base_kb = sorted(p.name for p in data_dir.glob("*.pdf"))  # 기존 코드와 동일하게 name 사용
        print("[DEBUG] KB base PDFs:", base_kb, flush=True)

        agent = SemiconductorRAGAgent(
            knowledge_pdf_paths=base_kb,
            force_reembed=False,  # 기존 임베딩 재사용
        )

        # 3) 새 문서들 일괄 추가 임베딩 (블로킹 방지)
        print("[DEBUG] Add new docs:", saved_paths, flush=True)
        await asyncio.to_thread(agent.add_new_documents, saved_paths)

        # 4) 결과 리턴
        payload = {
            "status": "ok",
            "message": f"{len(saved_paths)}개 문서를 추가했고 벡터DB에 반영했습니다.",
            "added_files": saved_paths,
        }
        if errors:
            payload["partial_errors"] = errors
        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

# 실행 명령어 예시: 순서대로 백엔드 띄운 후, 프론트엔드 띄우기, 현재 디렉토리 server_model 상위에서 실행 (상대 경로 . 사용)
# python -m uvicorn server_model.main:app --port 8001 --reload
# http://localhost:8001/static/index.html

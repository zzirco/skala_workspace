import os

# 기본 경로 설정 (환경 변수에서 가져오거나 기본값 사용)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server"))

# 상대 경로를 연결할 때 슬래시(/) 제거
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMAGE_DIR = os.path.join(BASE_DIR, "view-model-architecture")
MODEL_IMG_DIR = os.path.join(BASE_DIR, "model-images")

# 파일 경로 설정
DATA_PATH = os.path.join(UPLOAD_DIR, "IBM_2006-01-01_to_2018-01-01.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "temperature_prediction_model.keras")
MODEL_PLOT_PATH = os.path.join(IMAGE_DIR, "model.png")
MODEL_SHAPES_PLOT_PATH = os.path.join(IMAGE_DIR, "shapes/model_shapes.png")
PREDICTION_PLOT_PATH = os.path.join(IMAGE_DIR, "stock.png")

MODEL_SAVE_PATH_NEW = "./model/new_temperature_model.h5"
DATA_PATH_NEW = "./data/interpolated_temperature_1000.csv"

print("[DEBUG] BASE_DIR:", BASE_DIR)
print("[DEBUG] MODEL_SAVE_PATH:", MODEL_SAVE_PATH)
print("[DEBUG] EXISTS:", os.path.exists(MODEL_SAVE_PATH))
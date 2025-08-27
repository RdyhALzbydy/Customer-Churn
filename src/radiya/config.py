import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# مسارات المشروع
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# إعدادات البيانات
DATA_PATH = os.getenv("DATA_PATH", "data/raw/customer_churn_mini.json")
PROCESSED_DATA_PATH = DATA_DIR / "processed"

# إعدادات النماذج
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models/saved_models/")
SCALER_SAVE_PATH = MODEL_DIR / "scalers"

# إعدادات التدريب
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CHURN_METHODS = ['cancellation', 'downgrade', 'combined', 'inactivity']

# إعدادات MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "radiya_churn_prediction")

# إعدادات API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# إعدادات السجلات
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
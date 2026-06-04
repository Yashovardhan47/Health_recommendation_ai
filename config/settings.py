"""config/settings.py — Centralised configuration"""
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

APP_NAME    = "HealthAI"
APP_VERSION = "2.0.0"
SECRET_KEY  = "hc_secret_2024_xYzAb"

RF_N_ESTIMATORS       = 150
SVD_N_FACTORS         = 50
SVD_N_EPOCHS          = 20
DL_EMBED_DIM          = 16
DL_EPOCHS             = 15
RL_EPISODES           = 100
RL_LEARNING_RATE      = 0.1
HYBRID_DEFAULT_ALPHA  = 0.5
RANDOM_STATE          = 42
TEST_SIZE             = 0.20

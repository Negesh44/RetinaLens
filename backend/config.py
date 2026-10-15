"""Backend Configuration and Settings for RetinaLens API."""

from pathlib import Path
from typing import List
from ml.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    CLINICAL_DISCLAIMER,
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_UNCERTAINTY_THRESHOLD,
    REFERABLE_GRADE_THRESHOLD,
    ICDR_GRADES,
    ICDR_DESCRIPTIONS,
)

# Project paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Local secrets (e.g. GEMINI_API_KEY) from PROJECT_ROOT/.env; real environment variables win.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
STORAGE_DIR = PROJECT_ROOT / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
HEATMAPS_DIR = STORAGE_DIR / "heatmaps"
REPORTS_DIR = STORAGE_DIR / "reports"

# Ensure runtime directories exist
for d in [STORAGE_DIR, UPLOADS_DIR, HEATMAPS_DIR, REPORTS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = f"sqlite:///{STORAGE_DIR / 'retinalens.db'}"

# Model Configuration
DEFAULT_ONNX_PATH = CHECKPOINTS_DIR / "efficientnet_b0_retinalens.onnx"
DEFAULT_PYTORCH_PATH = CHECKPOINTS_DIR / "efficientnet_b0_regression_best.pth"
DEFAULT_BACKBONE = "efficientnet_b0"

# CORS configuration
CORS_ORIGINS: List[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]

# Max file size: 25 MB
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

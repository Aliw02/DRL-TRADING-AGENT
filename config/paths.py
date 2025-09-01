
# config/paths.py (KAGGLE-OPTIMIZED)
from pathlib import Path
import os

# --- Define Kaggle-specific directories ---
# The root directory for all outputs is the writable /kaggle/working/ directory
ROOT_DIR = Path(__name__).parent.parent.resolve()

# The data directory points to the read-only /kaggle/input/ directory
DATA_DIR = ROOT_DIR / "data"

print(f"Kaggle Root (Writable): {ROOT_DIR}")
print(f"Kaggle Data (Read-Only): {DATA_DIR}")

# --- Define output subdirectories ---
RESULTS_DIR = ROOT_DIR / "results"
WALK_FORWARD_DIR = RESULTS_DIR / "walk_forward"
FINAL_MODEL_DIR = RESULTS_DIR / "final_model_for_live"

# --- Define specific file paths ---
# Input data files (read from /kaggle/input)
TRAIN_DATA_FILE = DATA_DIR / "XAUUSDM1-FULL.csv"
BACKTEST_DATA_FILE = DATA_DIR / "XAUUSDM15-TEST-UNSEEN.csv"

# Generated data files (written to /kaggle/working/results)
PROCESSED_DATA_FILE = RESULTS_DIR / "processed_training_data.parquet"
FINAL_ENRICHED_DATA_FILE = RESULTS_DIR / "final_enriched_data.parquet"

# Final production model artifacts (written to /kaggle/working/results)
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "sac_finetuned_model.zip"
FINAL_SCALER_PATH = FINAL_MODEL_DIR / "final_agent_scaler.joblib"

# --- Create all necessary output directories ---
# We only create directories in the writable /kaggle/working area
for path in [RESULTS_DIR, WALK_FORWARD_DIR, FINAL_MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

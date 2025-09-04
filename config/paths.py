# config/paths.py (CORRECTED AND PROFESSIONAL VERSION)
from pathlib import Path
import os

# --- Define project directories ---
# The root directory for the project code
ROOT_DIR = Path(__file__).parent.parent.resolve()
# The root directory for data and results on your Google Drive
DRIVE_DIR = Path("/content/drive/MyDrive").resolve()

# --- Define data and results paths ---
DATA_DIR = DRIVE_DIR / "data"
RESULTS_DIR = DRIVE_DIR / "results"
WALK_FORWARD_DIR = RESULTS_DIR / "walk_forward"
FINAL_MODEL_DIR = RESULTS_DIR / "final_model_for_live"

print(f"Project Root (Code): {ROOT_DIR}")
print(f"Drive Root (Data/Results): {DRIVE_DIR}")

# --- Define specific file paths ---
# Input data files
TRAIN_DATA_FILE = DATA_DIR / "XAUUSDM1-FULL.csv"
BACKTEST_DATA_FILE = DATA_DIR / "XAUUSDM15-TEST-UNSEEN.csv"
BACKTEST_M1TF_DATA_FILE = DATA_DIR / "XAUUSDM1-TEST-UNSEEN.csv"

# Generated data files
PROCESSED_DATA_FILE = RESULTS_DIR / "processed_training_data.parquet"
FINAL_ENRICHED_DATA_FILE = RESULTS_DIR / "final_enriched_data.parquet"

# Final production model artifacts
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "sac_finetuned_model.zip"
FINAL_SCALER_PATH = FINAL_MODEL_DIR / "final_agent_scaler.joblib"

# --- Create all necessary output directories ---
# This loop now correctly and safely creates the directories if they don't exist.
for path in [RESULTS_DIR, WALK_FORWARD_DIR, FINAL_MODEL_DIR]:
    # The 'parents=True' argument creates parent directories if needed.
    # The 'exist_ok=True' argument prevents an error if the directory already exists.
    path.mkdir(parents=True, exist_ok=True)
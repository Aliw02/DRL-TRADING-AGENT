# config/paths.py (UPDATED)
from pathlib import Path
import os
try: ROOT_DIR = Path(__file__).parent.parent.resolve()
except NameError: ROOT_DIR = Path(os.getcwd()).resolve()
print(f"Project Root Directory: {ROOT_DIR}")
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
# --- NEW: Walk-Forward Directory ---
WALK_FORWARD_DIR = RESULTS_DIR / "walk_forward"
FINAL_MODEL_DIR = RESULTS_DIR / "final_model_for_live"
# Data files
TRAIN_DATA_FILE = DATA_DIR / "XAUUSDM1-FULL.csv"
BACKTEST_DATA_FILE = DATA_DIR / "XAUUSDM15-TEST-UNSEEN.csv"
PROCESSED_DATA_FILE = RESULTS_DIR / "processed_training_data.parquet"
FINAL_ENRICHED_DATA_FILE = RESULTS_DIR / "final_enriched_data.parquet"
# Final production artifacts
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "sac_finetuned_model.zip"
FINAL_SCALER_PATH = FINAL_MODEL_DIR / "final_agent_scaler.joblib"

for path in [DATA_DIR, RESULTS_DIR, WALK_FORWARD_DIR, FINAL_MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
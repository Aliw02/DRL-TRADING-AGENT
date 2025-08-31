from pathlib import Path
import os
ROOT_DIR = Path('/kaggle/working/DRL-TRADING-AGENT').resolve()
DATA_DIR = Path('/kaggle/input/{kaggle_dataset_folder}').resolve()
OUTPUT_DIR = Path('/kaggle/working').resolve()
RESULTS_DIR = ROOT_DIR / "results"
print(f"Project Root: {{ROOT_DIR}}")
print(f"Data Input (Read-Only): {{DATA_DIR}}")
print(f"Data/Results Output (Writable): {{OUTPUT_DIR}}")
TRAIN_DATA_FILE = DATA_DIR / "XAUUSDM1-FULL.csv"
BACKTEST_DATA_FILE = DATA_DIR / "XAUUSDM15-TEST-UNSEEN.csv"
PROCESSED_DATA_FILE = RESULTS_DIR / "processed_training_data.parquet"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
WALK_FORWARD_DIR = RESULTS_DIR / "walk_forward"
FINAL_MODEL_DIR = RESULTS_DIR / "final_model_for_live"
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "finetuned_model.zip"
FINAL_SCALER_PATH = FINAL_MODEL_DIR / "final_robust_scaler.joblib"
for path in [RESULTS_DIR, LOGS_DIR, PLOTS_DIR, WALK_FORWARD_DIR, FINAL_MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
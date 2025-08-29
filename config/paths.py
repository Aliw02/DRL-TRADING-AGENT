# config/paths.py (NEW FILE)

from pathlib import Path
import os

# --- DYNAMICALLY LOCATE THE PROJECT ROOT ---
# This script will navigate up the directory tree from the current file's location
# until it finds a file or directory that signals the project root (e.g., '.gitignore' or 'requirements.txt').
# This makes the paths robust and independent of where the script is executed from.
try:
    ROOT_DIR = Path(__file__).parent.parent.resolve()
except NameError:
    # If __file__ is not defined (e.g., in an interactive environment),
    # fall back to the current working directory.
    ROOT_DIR = Path(os.getcwd()).resolve()

print(f"Project Root Directory determined as: {ROOT_DIR}")

# --- DATA FOLDER ---
DATA_DIR = ROOT_DIR / "data"
# Define your specific data files here
TRAIN_DATA_FILE = DATA_DIR / "XAUUSDM1-FULL.csv"  # Example training data file
BACKTEST_DATA_FILE = DATA_DIR / "XAUUSDM15-TEST-UNSEEN.csv" # Example backtest data file

# --- RESULTS FOLDER ---
# This is the main output directory for all artifacts
RESULTS_DIR = ROOT_DIR / "results"

# Subdirectories for specific artifacts
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
WALK_FORWARD_DIR = RESULTS_DIR / "walk_forward"

# --- SPECIFIC MODEL AND ARTIFACT PATHS ---
# Paths for the final, fine-tuned model ready for live trading
FINAL_MODEL_DIR = MODELS_DIR / "final_model_for_live"
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "finetuned_model.zip"
FINAL_SCALER_PATH = FINAL_MODEL_DIR / "final_robust_scaler.joblib"

# --- CONFIGURATION FILES ---
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"

# --- Create directories if they don't exist ---
# This ensures that the code doesn't fail if a directory is missing.
for path in [DATA_DIR, RESULTS_DIR, LOGS_DIR, PLOTS_DIR, MODELS_DIR, WALK_FORWARD_DIR, FINAL_MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
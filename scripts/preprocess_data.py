import pandas as pd
import sys
import os
import traceback

# Add the project root to the path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_transformation import DataTransformer
from utils.logger import setup_logging, get_logger
from config import paths # Import the corrected paths

def main():
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING FINAL DATA PREPROCESSING (from clean CSV)")
    logger.info("="*80)

    try:
        transformer = DataTransformer()
        
        # --- 1. READ from the correct INPUT path ---
        logger.info(f"Loading raw training data from: {{paths.TRAIN_DATA_FILE}}")
        processed_df = transformer.load_and_preprocess_data(file_path=str(paths.TRAIN_DATA_FILE))
        
        # --- 2. WRITE to the correct OUTPUT path ---
        # paths.PROCESSED_DATA_FILE correctly points to /kaggle/working/
        output_path = paths.PROCESSED_DATA_FILE
        
        logger.info(f"Saving final processed data to Parquet format at: {{output_path}}")
        processed_df.to_parquet(output_path)
        logger.info(f"✅ Final data successfully preprocessed and saved as Parquet. Shape: {{processed_df.shape}}")

    except Exception as e:
        logger.error(f"An error occurred during final preprocessing: {{e}}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
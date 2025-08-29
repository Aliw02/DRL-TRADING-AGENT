# scripts/preprocess_data.py (NEW SCRIPT)

import pandas as pd
import sys
import os

# -- Add project root to path --
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_transformation import DataTransformer
from config import paths
from utils.logger import setup_logging, get_logger

def main():
    """
    This script performs a one-time, memory-intensive preprocessing task.
    It loads the raw, large CSV data, engineers all features, and saves
    the result to a highly efficient Parquet file for faster, memory-light training.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING ONE-TIME DATA PREPROCESSING")
    logger.info(f"Loading raw data from: {paths.TRAIN_DATA_FILE}")
    logger.info("This may take a significant amount of time and RAM...")
    logger.info("="*80)

    try:
        transformer = DataTransformer()
        # This is the step that consumes a lot of RAM
        processed_df = transformer.load_and_preprocess_data(file_path=str(paths.TRAIN_DATA_FILE))

        # Define the output path for the new, efficient file
        output_path = paths.DATA_DIR / "processed_training_data.parquet"
        
        logger.info(f"Preprocessing complete. Shape of the final data: {processed_df.shape}")
        logger.info(f"Saving processed data to Parquet format at: {output_path}")

        # Save to Parquet format
        processed_df.to_parquet(output_path)

        logger.info("✅ Data successfully preprocessed and saved as Parquet.")
        logger.info("You can now run the main training script.")
        logger.info("="*80)

    except MemoryError:
        logger.error("FATAL: Ran out of system RAM during preprocessing.")
        logger.error("Suggestion: Run this script on a machine with more RAM (e.g., a high-RAM Kaggle/Colab session or a larger cloud VM).")
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}", exc_info=True)

if __name__ == "__main__":
    # You need to have pyarrow installed: pip install pyarrow fastparquet
    main()
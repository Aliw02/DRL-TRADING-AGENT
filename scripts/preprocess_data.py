# scripts/preprocess_data.py (FOCUSED AND FINAL VERSION)
import sys
import os

# -- Add project root to path to allow imports from other directories --
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_transformation import DataTransformer
from config import paths
from utils.logger import setup_logging, get_logger

def run_preprocessing():
    """
    This script now performs a single, focused task that serves as the
    first step in the main pipeline:
    - Loads the raw CSV data.
    - Calculates all advanced technical indicators needed for subsequent stages.
    - Saves the result to an intermediate Parquet file, which will be used for
      both regime detection and as a base for the final enriched dataset.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING INITIAL FEATURE ENGINEERING PIPELINE")
    logger.info("="*80)

    try:
        transformer = DataTransformer()
        
        # This function loads the raw CSV and then calls `calculate_all_indicators`
        # to perform the heavy feature engineering.
        base_processed_df = transformer.load_and_preprocess_data(file_path=str(paths.TRAIN_DATA_FILE), timeframe='15min')

        # The output of this script is the foundational dataset upon which all
        # other pipeline steps will be built.
        intermediate_path = paths.PROCESSED_DATA_FILE
        base_processed_df.to_parquet(intermediate_path)
        
        logger.info(f"✅ Initial feature engineering complete.")
        logger.info(f"Processed data with all indicators saved to: {intermediate_path}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"An error occurred during initial preprocessing: {e}", exc_info=True)
        # Re-raise the exception to halt the main pipeline if this critical step fails
        raise

if __name__ == "__main__":
    run_preprocessing()
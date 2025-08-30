# utils/data_transformation.py (FINAL CORRECTED VERSION)

import pandas as pd
from utils.logger import get_logger
from utils.custom_indicators import calculate_all_indicators

logger = get_logger(__name__)

class DataTransformer:
    """
    Loads a CLEAN, comma-separated data file and engineers all features.
    """
    def load_and_preprocess_data(self, file_path):
        try:
            logger.info(f"Loading CLEAN, COMMA-SEPARATED data from: {file_path}")
            
            # --- Define column names for the clean format ---
            column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # --- Read the data as a standard CSV file ---
            # This now matches the output of our cleansing script.
            df = pd.read_csv(
                file_path, 
                sep=',', # The separator is now a comma
                header=None,
                names=column_names,
                parse_dates=['timestamp'] # Tell pandas to treat the first column as a date
            )
            
            # --- Set the timestamp as the DataFrame index ---
            df.set_index('timestamp', inplace=True)
            
            # --- Cleanse the data ---
            original_rows = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            df = df[df.index.notna()]
            new_rows = len(df)
            if original_rows > new_rows:
                logger.warning(f"Removed {original_rows - new_rows} rows with invalid data.")

            logger.info("--> STEP 1.1: Clean data loaded successfully.")

            # --- Calculate all features ---
            logger.info("--> STEP 1.2: Starting advanced feature engineering...")
            df_with_features = calculate_all_indicators(df)
            logger.info("--> STEP 1.3: All features calculated.")
            
            processed_df = df_with_features.copy()
            print("Before dropping NA: " + str(processed_df.shape))
            processed_df.dropna(inplace=True)
            print("After dropping NA: " + str(processed_df.shape))

            logger.info(f"Data preprocessing complete. Final shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error during data loading and preprocessing: {e}", exc_info=True)
            raise
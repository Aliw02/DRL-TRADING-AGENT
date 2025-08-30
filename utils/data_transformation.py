# utils/data_transformation.py (ROBUST CLEANUP VERSION)

import pandas as pd
from utils.logger import get_logger
from utils.custom_indicators import calculate_all_indicators

logger = get_logger(__name__)

class DataTransformer:
    def load_and_preprocess_data(self, file_path):
        try:
            logger.info(f"Loading CLEAN, COMMA-SEPARATED data from: {file_path}")
            
            df = pd.read_csv(
                file_path,
                sep=',',
                header=0,
                parse_dates=['timestamp'],
                date_format='%Y-%m-%d %H:%M'
            )
            df.set_index('timestamp', inplace=True)

            original_rows = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            df = df[df.index.notna()]
            new_rows = len(df)
            if original_rows > new_rows:
                logger.warning(f"Removed {original_rows - new_rows} rows with invalid initial data.")

            logger.info("--> STEP 1.1: Clean data loaded successfully.")

            logger.info("--> STEP 1.2: Starting advanced feature engineering...")
            df_with_features = calculate_all_indicators(df)
            logger.info("--> STEP 1.3: All features calculated.")

            processed_df = df_with_features.copy()
            
            # --- NEW, MORE ROBUST CLEANUP LOGIC ---
            logger.info(f"Data shape before smart cleanup: {processed_df.shape}")

            # 1. Drop any COLUMN that is entirely empty (all NaN)
            cols_before = set(processed_df.columns)
            processed_df.dropna(axis='columns', how='all', inplace=True)
            cols_after = set(processed_df.columns)
            if len(cols_before) > len(cols_after):
                removed_cols = cols_before - cols_after
                logger.warning(f"Removed completely empty columns: {removed_cols}")
            
            # 2. Now, drop any remaining ROW that has NaNs (usually the first few rows)
            rows_before = len(processed_df)
            processed_df.dropna(axis='rows', how='any', inplace=True)
            rows_after = len(processed_df)
            
            logger.info(f"Removed {rows_before - rows_after} initial rows with NaN values.")
            
            logger.info(f"Data preprocessing complete. Final shape: {processed_df.shape}")
            return processed_df

        except Exception as e:
            logger.error(f"Error during data loading and preprocessing: {e}", exc_info=True)
            raise
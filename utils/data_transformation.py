# utils/data_transformation.py (FINAL, CORRECTED & PROFESSIONAL VERSION)

import pandas as pd
from utils.logger import get_logger
from utils.custom_indicators import calculate_all_indicators

logger = get_logger(__name__)

class DataTransformer:
    def load_and_preprocess_data(self, file_path: str, timeframe: str = '1min'):
        """
        Loads and preprocesses financial data with professional resampling logic.

        Args:
            file_path (str): The path to the raw 1-minute CSV data file.
            timeframe (str): The target timeframe to resample the data to 
                             (e.g., '15min', '1h', '4h'). Defaults to '1min'.

        Returns:
            pd.DataFrame: A fully processed DataFrame with calculated indicators
                          at the specified timeframe.
        """
        try:
            logger.info(f"Loading 1-minute raw data from: {file_path}")
            
            df = pd.read_csv(
                file_path,
                sep=',',
                header=0,
                parse_dates=['timestamp'],
                date_format='%Y-%m-%d %H:%M'
            )
            df.set_index('timestamp', inplace=True)

            # --- Initial Data Cleaning ---
            original_rows = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
            df = df[df.index.notna()]
            new_rows = len(df)
            if original_rows > new_rows:
                logger.warning(f"Removed {original_rows - new_rows} rows with invalid initial data.")

            logger.info("--> STEP 1.1: Raw data loaded successfully.")

            # --- PROFESSIONAL RESAMPLING LOGIC ---
            if timeframe != '1min':
                logger.info(f"--> STEP 1.2: Resampling 1-minute data to {timeframe} timeframe...")
                
                # Define the aggregation rules for resampling OHLCV data
                aggregation_rules = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                
                # Resample the dataframe
                df_resampled = df.resample(timeframe).agg(aggregation_rules)
                
                # Drop any rows that might be empty after resampling (e.g., weekends)
                df_resampled.dropna(inplace=True)
                
                logger.info(f"Resampling complete. New shape: {df_resampled.shape}")
                df = df_resampled
            else:
                logger.info("--> STEP 1.2: Skipping resampling for 1min timeframe.")

            # --- Feature Engineering ---
            logger.info("--> STEP 1.3: Starting advanced feature engineering...")
            df_with_features = calculate_all_indicators(df)
            logger.info("--> STEP 1.4: All features calculated.")

            # --- Final Cleanup Logic ---
            processed_df = df_with_features.copy()
            logger.info(f"Data shape before final cleanup: {processed_df.shape}")

            # Drop any column that is entirely empty
            processed_df.dropna(axis='columns', how='all', inplace=True)
            
            # Drop any remaining row that has NaNs (usually the first few rows from indicator calculation)
            rows_before = len(processed_df)
            processed_df.dropna(axis='rows', how='any', inplace=True)
            rows_after = len(processed_df)
            logger.info(f"Removed {rows_before - rows_after} initial rows with NaN values.")
            
            logger.info(f"Data preprocessing complete. Final shape: {processed_df.shape}")
            return processed_df

        except Exception as e:
            logger.error(f"Error during data loading and preprocessing: {e}", exc_info=True)
            raise

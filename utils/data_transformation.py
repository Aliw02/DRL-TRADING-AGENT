# utils/data_transformation.py (MODERNIZED VERSION)

import pandas as pd
from utils.logger import get_logger
from utils.custom_indicators import calculate_all_indicators

logger = get_logger(__name__)

class DataTransformer:
    def load_and_preprocess_data(self, file_path):
        try:
            logger.info(f"Loading data from: {file_path}")
            # Assuming Tickstory format
            column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            df = pd.read_csv(file_path, sep=',', header=None, names=column_names, 
                             dtype={'date': str, 'time': str})
            
            # Combine date + time into a single datetime index
            df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y%m%d %H:%M:%S')
            df.set_index('time', inplace=True)
            df.drop(columns=['date'], inplace=True)

            logger.info("--> STEP 1.1: Raw data loaded.")

            # --- Calculate all base and advanced features ---
            logger.info("--> STEP 1.2: Starting advanced feature engineering...")
            df_with_features = calculate_all_indicators(df)
            logger.info("--> STEP 1.3: All features calculated.")
            
            # --- Final Preparation ---
            # We keep the 'close' price for the environment, but it won't be a feature.
            # All other calculated columns will be used as features.
            processed_df = df_with_features.copy()
            
            processed_df.dropna(inplace=True)
            processed_df.reset_index(drop=True, inplace=True) 
            
            logger.info(f"Data preprocessing complete. Final shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error during data loading and preprocessing: {e}")
            raise
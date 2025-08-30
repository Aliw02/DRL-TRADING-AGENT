# utils/data_transformation.py (UPDATED FOR WHITESPACE-SEPARATED FORMAT)

import pandas as pd
from utils.logger import get_logger
from utils.custom_indicators import calculate_all_indicators

logger = get_logger(__name__)

class DataTransformer:
    """
    Handles loading data from a whitespace-separated format and
    engineering all necessary features for the trading agent.
    """
    def load_and_preprocess_data(self, file_path):
        """
        Loads data with a format like: 'YYYY-MM-DD HH:MM OPEN HIGH LOW CLOSE VOLUME'
        separated by one or more spaces/tabs.
        """
        try:
            logger.info(f"Loading whitespace-separated data from: {file_path}")
            
            # --- Define column names for the new format ---
            column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            
            # --- Read the data using a regular expression for the separator ---
            # '\s+' tells pandas to treat one or more spaces/tabs as a single separator.
            df = pd.read_csv(
                file_path, 
                sep=r'\s+', # This is the key change
                header=None,
                names=column_names,
                dtype={'date': str, 'time': str}
            )

            logger.info("--> STEP 1.1: Raw data loaded successfully.")

            # --- Combine date and time columns and set as index ---
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            df.set_index('timestamp', inplace=True)
            df.drop(columns=['date', 'time'], inplace=True)
            
            # --- Cleanse the data ---
            original_rows = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            df = df[df.index.notna()]
            new_rows = len(df)
            if original_rows > new_rows:
                logger.warning(f"Removed {original_rows - new_rows} rows with invalid timestamps or data.")

            logger.info("--> STEP 1.2: Timestamp index created and cleansed.")

            # --- Calculate all features ---
            logger.info("--> STEP 1.3: Starting advanced feature engineering...")
            df_with_features = calculate_all_indicators(df)
            logger.info("--> STEP 1.4: All features calculated.")
            
            processed_df = df_with_features.copy()
            processed_df.dropna(inplace=True)

            logger.info(f"Data preprocessing complete. Final shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error during data loading and preprocessing: {e}", exc_info=True)
            raise
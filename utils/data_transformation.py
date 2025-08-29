import pandas as pd
from sklearn.preprocessing import StandardScaler
from config.init import config
from utils.logger import get_logger
from utils.custom_indicators import calculate_all_indicators

logger = get_logger(__name__)

class DataTransformer:
    """
    Handles loading data and calculating all features.
    The splitting and scaling is now handled by the walk-forward training script.
    """
    
    def __init__(self):
        # The scaler is no longer needed here, it will be created for each split
        self.feature_columns = config.get('data.feature_columns', [])
    
    def load_and_preprocess_data(self, file_path=None):
        """
        Loads the raw data, calculates all indicators and features,
        and returns a single, complete DataFrame.
        """
        try:
            file_path = file_path or config.get('data.file_path')
            logger.info(f"Loading data from: {file_path}")
            
            # --- تحميل البيانات ---
            # This is a more robust way to load the CSV, assuming it has a header.
            # If your file is truly tab-separated with no header, revert to the old method.
            logger.info(f"Reading data from source: {file_path}")
            is_tick_story_source_data = True
            if is_tick_story_source_data:
                # Sample record: 20160418,00:00:00,1236.989,1237.27,1236.789,1236.999,154,154,259
                # Columns: date, time, open, high, low, close, tick_volume, volume, spread
                column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'volume', 'spread']
                df = pd.read_csv(file_path, sep=',', header=None, names=column_names, dtype={'date': str, 'time': str})
                # Combine date + time into a single datetime index
                df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y%m%d %H:%M:%S', errors='coerce')
                df.drop(columns=['date'], inplace=True)
                df.set_index('time', inplace=True)
                # Ensure a single 'volume' column for downstream processing: prefer 'volume', fallback to 'tick_volume'
                if 'volume' not in df.columns and 'tick_volume' in df.columns:
                    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
                elif 'tick_volume' in df.columns and 'volume' in df.columns:
                    df.drop(columns=['tick_volume'], inplace=True)
            else:
                column_names = ['time', 'open', 'high', 'low', 'close', 'volume']
                df = pd.read_csv(file_path, sep='\t', header=None, names=column_names, index_col='time', parse_dates=True)

            df = df[['open', 'high', 'low', 'close', 'volume']]
            logger.info("--> STEP 1.1: CSV loaded.")

            # Validate required columns for indicators
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Source data is missing one or more required columns: 'open', 'high', 'low', 'close'.")

            # --- حساب كل المؤشرات ---
            logger.info("--> STEP 1.2: Calculating all indicators and features...")
            df_with_indicators = calculate_all_indicators(df)
            logger.info("--> STEP 1.3: All indicators calculated.")
            
            # --- دمج سعر الإغلاق مع الميزات ---
            # We need the 'close' price for the environment, but it shouldn't be scaled as a feature.
            # First, select only the feature columns specified in the config.
            processed_df = df_with_indicators[self.feature_columns].copy()
            
            # Now, add the original 'close' price to the processed DataFrame.
            processed_df['close'] = df['close']
            
            # --- تنظيف نهائي للبيانات ---
            # It's crucial to drop NaNs that were created by the indicators.
            processed_df.dropna(inplace=True)
            processed_df.reset_index(drop=True, inplace=True) # Reset index for easy slicing later
            
            logger.info(f"Data preprocessing complete. Final shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error during data loading and preprocessing: {e}")
            raise
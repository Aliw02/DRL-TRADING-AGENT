# live_trading/mt5_connector.py
# SECURE CONNECTOR AND DATA FETCHER FOR METATRADER 5

import MetaTrader5 as mt5
import pandas as pd
import sys
import os

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.logger import get_logger

logger = get_logger(__name__)

def connect_mt5(login, password, server):
    """
    Connects to the MT5 terminal and logs in using provided credentials.
    """
    if not mt5.initialize():
        logger.critical(f"MT5 initialize() failed, error: {mt5.last_error()}")
        return False
        
    authorized = mt5.login(login, password, server)
    if not authorized:
        logger.critical(f"MT5 Login failed for account {login}, error: {mt5.last_error()}")
        mt5.shutdown()
        return False

    logger.info("MT5 connection and login successful.")
    return True

def get_mt5_data(symbol, timeframe, num_bars):
    """
    Fetches historical bar data from MT5 and standardizes the volume column name.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to fetch data for {symbol}. No rates returned.")
            return None
        
        df = pd.DataFrame(rates)
        
        # Standardize the volume column: MT5 uses 'tick_volume'. Our pipeline uses 'volume'.
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        else:
            df['volume'] = 0 # Fallback if no volume data is present
            
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"Error occurred while fetching MT5 data: {e}", exc_info=True)
        return None
# mt5_connector.py
# mt5_connector.py (No major changes, just ensure it can handle larger requests)
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
from utils.logger import setup_logging

logger = setup_logging()

def connect_mt5(login, password, server):
    """
    يتصل بمنصة MT5 ويسجل الدخول باستخدام بيانات الاعتماد.
    """
    if not mt5.initialize():
        print("initialize() failed, error: ", mt5.last_error())
        return False
        
    authorized = mt5.login(login, password, server)
    if not authorized:
        print("Login failed, error: ", mt5.last_error())
        mt5.shutdown()
        return False

    print("Connected and logged in successfully!")
    return True

def get_mt5_data(symbol, timeframe, num_bars):
    """
    يحصل على البيانات التاريخية من MT5 ويقوم بتوحيد اسم عمود الحجم.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            print("Failed to get data.")
            return None
        
        df = pd.DataFrame(rates)
        
        # --- هذا هو السطر الذي يحل المشكلة ---
        # MT5 provides 'tick_volume'. We rename it to 'volume' to match the
        # column name used throughout the entire training pipeline.
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        else:
            # As a fallback, create a volume column of zeros if not present
            df['volume'] = 0
            
        # The 'time' column is essential for resampling
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        # We only need the OHLCV columns for our model
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"Error occurred while fetching data: {e}")
        return None
        

def get_open_position(symbol):
    """Returns the first open position object for a given symbol, or None."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is not None and len(positions) > 0:
            return positions[0]
        return None
    except Exception as e:
        logger.error(f"Error getting open position: {e}")
        return None

def open_trade(action, symbol, lot_size, stop_loss, take_profit):
    """Opens a new trade with specified SL and TP."""
    trade_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if trade_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": trade_type,
        "price": price,
        "sl": round(stop_loss, 2),
        "tp": round(take_profit, 2),
        "deviation": 20,
        "magic": 234000,
        "comment": "DRL Agent Entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    logger.info(f"Sending OPEN request: {action} {lot_size} {symbol} @ {price} SL:{request['sl']} TP:{request['tp']}")
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Failed to open trade, retcode: {result.retcode}, reason: {result.comment}")
        return None
    
    logger.info(f"Trade opened successfully. Position ticket: {result.order}")
    return result

def close_trade(position):
    """Closes an existing position based on the position object."""
    trade_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": trade_type,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "DRL Agent Exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    logger.info(f"Sending CLOSE request for ticket: {position.ticket}")
    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Failed to close trade, retcode: {result.retcode}, reason: {result.comment}")
        return False

    logger.info(f"Trade closed successfully for ticket: {position.ticket}")
    return True
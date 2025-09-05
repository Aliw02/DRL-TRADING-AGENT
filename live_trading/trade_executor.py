# live_trading/trade_executor.py
# TRADE EXECUTION AND POSITION MANAGEMENT UNIT

import MetaTrader5 as mt5
import sys
import os

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.logger import get_logger

logger = get_logger(__name__)

def get_open_position(symbol):
    """ Returns the first open position object for a given symbol, or None. """
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is not None and len(positions) > 0:
            return positions[0]
        return None
    except Exception as e:
        logger.error(f"Error getting open position: {e}")
        return None

def close_position(position):
    """ Closes a given position with a market order. """
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(position.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "DRL Agent Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Failed to close position {position.ticket}, retcode: {result.retcode}")
    else:
        logger.info(f"Position {position.ticket} closed successfully.")
    return result

def execute_trade(symbol: str, action_value: float, lot_size: float = 0.01):
    """
    Executes a buy, sell, or hold action based on the model's continuous signal.
    - action_value > 0.5:  Enter/Maintain BUY
    - action_value < -0.5: Enter/Maintain SELL (Not implemented for simplicity)
    - -0.5 < action_value < 0.5: Close any open position / Hold
    """
    open_position = get_open_position(symbol)

    # --- Close/Hold Logic ---
    if -0.5 <= action_value <= 0.5:
        if open_position:
            logger.info(f"Signal is neutral ({action_value:.2f}). Closing open position.")
            close_position(open_position)
        else:
            logger.info(f"Signal is neutral ({action_value:.2f}). Holding cash position.")
        return

    # --- Buy Logic ---
    if action_value > 0.5:
        if open_position and open_position.type == mt5.ORDER_TYPE_BUY:
            logger.info("Signal is BUY and a BUY position is already open. Maintaining position.")
            return
        if open_position and open_position.type == mt5.ORDER_TYPE_SELL:
            logger.info("Signal is BUY, but SELL position is open. Closing first.")
            close_position(open_position)
            # We wait for the next cycle to open a new position to avoid thrashing
            return

        # Open new BUY position
        price = mt5.symbol_info_tick(symbol).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY, "price": price, "deviation": 20, "magic": 234000,
            "comment": "DRL Agent Entry", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to open BUY order, retcode: {result.retcode}")
        else:
            logger.info(f"BUY order sent successfully for {lot_size} lots.")

    # --- Sell Logic (can be expanded similarly) ---
    elif action_value < -0.5:
        logger.info("SELL signal received. Current simplified logic is to close any BUY positions.")
        if open_position and open_position.type == mt5.ORDER_TYPE_BUY:
            close_position(open_position)
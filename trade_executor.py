# trade_executor.py (Updated)
import MetaTrader5 as mt5

def get_open_position(symbol):
    """
    Returns the first open position for a given symbol, or None if no position is open.
    """
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is not None and len(positions) > 0:
            return positions[0]
        return None
    except Exception as e:
        print(f"Error getting open position: {e}")
        return None

def close_position(position):
    """
    Closes a given position.
    """
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "deviation": 20,
        "magic": 234000,
        "comment": "DRL Bot - Close Position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close position, retcode: {result.retcode}")
    else:
        print("Position closed successfully.")
    return result

def execute_trade(symbol, action, lot_size=0.01):
    """
    Executes a buy, sell, or close trade based on the model's action and the current open position.
    """
    open_position = get_open_position(symbol)

    if open_position:
        # If there's an open position, check if the action is to close it
        if open_position.type == mt5.ORDER_TYPE_BUY and action == 2: # Model recommends SELL
            print("Model recommends SELL. Closing open BUY position.")
            close_position(open_position)
            return None
        elif open_position.type == mt5.ORDER_TYPE_SELL and action == 1: # Model recommends BUY
            print("Model recommends BUY. Closing open SELL position.")
            close_position(open_position)
            return None
        else:
            print("Model recommends HOLD or the same position type. No action taken.")
            return None
    else:
        # If no open position, check if the action is to open a new one
        if action == 1:  # Buy
            order_type = mt5.ORDER_TYPE_BUY
            print("Model recommends BUY. Opening new position.")
        elif action == 2:  # Sell
            order_type = mt5.ORDER_TYPE_SELL
            print("Model recommends SELL. Opening new position.")
        else: # Hold
            print("Model recommends HOLD. No action taken.")
            return None

        # Execute the order to open a new position
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "DRL Trading Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to send order, retcode: {result.retcode}")
        else:
            print("Order sent successfully.")

        return result
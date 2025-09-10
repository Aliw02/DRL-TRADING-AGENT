# scripts/backtest_hierarchical.py
# CE-ALIGNED HIERARCHICAL BACKTESTING SIMULATOR (CHAMPIONSHIP EDITION) - V7 (Trade Logging Enabled)
import pandas as pd
import joblib
import os
import sys
import numpy as np
import math
import talib as ta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import SAC
from config import paths
from config.init import Config
from utils.logger import setup_logging, get_logger
from utils.metrics import calculate_performance_metrics
from utils.accelerator import DEVICE
from utils.data_transformation import DataTransformer

def calculate_commission(lot_size: float, hour: int) -> float:
    """
    Calculates trading commission based on the time of day and lot size.
    The cost is based on a rate per 0.01 lots.
    """
    if 8 <= hour < 21: # Day session: 8 AM to 9 PM (21:00)
        cost_per_001_lot = 0.20
    else: # Night session
        cost_per_001_lot = 0.35
    
    num_blocks = abs(lot_size) / 0.01
    return num_blocks * cost_per_001_lot


def run_hierarchical_backtest(backtest_df: pd.DataFrame = None, results_suffix: str = ""):
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"🚀 INITIATING FINAL BACKTEST WITH TRADE LOGGING...")

    agent_config = Config()
    
    use_ema_filter = True
    use_time_filter = True
    trading_start_hour = 8 
    trading_end_hour = 21    
    
    atr_multiplier = 4.0
    
    base_lot_size = 0.01
    lot_step_per_profit = 0.01
    profit_threshold_for_step = 100.0
    min_lot_size = 0.01
    max_lot_size = 10.0
    
    dynamic_profit_enabled = True

    logger.info(f"Filters: EMA Cross = {use_ema_filter}, Time = {use_time_filter} ({trading_start_hour}:00-{trading_end_hour}:00)")
    logger.info(f"Lot Sizing: Base Lot = {base_lot_size}, Step = {lot_step_per_profit} per ${profit_threshold_for_step} profit.")
    
    try:
        squad = {'specialists': {}}
        model_dir = paths.FINAL_MODEL_DIR
        
        specialist_map = {1: "bullish", -1: "bearish"}
        for state_id, state_name in specialist_map.items():
            specialist_dir = model_dir / f"specialist_{state_name}"
            model_path, scaler_path = specialist_dir / "champion_model.zip", specialist_dir / "champion_scaler.joblib"
            if model_path.exists() and scaler_path.exists():
                squad['specialists'][state_id] = {'model': SAC.load(model_path, device=DEVICE), 'scaler': joblib.load(scaler_path)}
                logger.info(f"-> CHAMPION Specialist for {state_name.upper()} state loaded.")
            else: 
                logger.warning(f"Champion model for {state_name.upper()} not found.")

        if not squad['specialists']:
            logger.error("No champion specialist models were loaded. Cannot proceed.")
            return
        
        # --- MODIFICATION START: Use provided DataFrame if available ---
        if backtest_df is None:
            logger.info("STEP 1: Loading and processing unseen test data...")
            transformer = DataTransformer()
            backtest_df = transformer.load_and_preprocess_data(file_path=str(paths.BACKTEST_DATA_FILE), timeframe="15min")
        else:
            logger.info("STEP 1: Using pre-loaded DataFrame for backtest (Stress Test Mode).")
        # --- MODIFICATION END ---

        logger.info("Calculating external EMA filters...")
        backtest_df['ema_fast'] = ta.EMA(backtest_df['close'], timeperiod=50)
        backtest_df['ema_slow'] = ta.EMA(backtest_df['close'], timeperiod=200)
        backtest_df.ffill(inplace=True)

        starting_balance = agent_config.get('environment.initial_balance')
        initial_balance = starting_balance
        equity = initial_balance
        equity_curve = [initial_balance]
        trades = []
        open_position = None
        sequence_length = agent_config.get('environment.sequence_length', 70)
        
        last_weekly_equity, last_monthly_equity = initial_balance, initial_balance
        
        logger.info("STEP 2: Starting simulation loop...")
        for i in range(sequence_length, len(backtest_df)):
            current_bar, previous_bar = backtest_df.iloc[i], backtest_df.iloc[i-1]
            
            if open_position:
                unrealized_pnl = (current_bar['close'] - open_position['entry_price']) * (open_position['lot_size'] / 0.01)
                equity = open_position['balance_at_entry'] + unrealized_pnl
                
                exit_reason, exit_price = None, current_bar['close']
                
                if (open_position['type'] == 'BUY' and not dynamic_profit_enabled) and action_value < -0.5 or \
                   (open_position['type'] == 'SELL' and not dynamic_profit_enabled) and action_value > 0.5:
                    exit_reason = "Reversal Signal"
                    exit_price = current_bar['close']

                elif (open_position['type'] == 'BUY' and current_bar['low'] <= open_position['stop_loss']) or \
                   (open_position['type'] == 'SELL' and current_bar['high'] >= open_position['stop_loss']):
                    exit_reason, exit_price = "Stop-Loss", open_position['stop_loss']
                elif not dynamic_profit_enabled and ((open_position['type'] == 'BUY' and current_bar['high'] >= open_position['take_profit']) or (open_position['type'] == 'SELL' and current_bar['low'] <= open_position['take_profit'])):
                    exit_reason, exit_price = "Take-Profit", open_position['take_profit']
                elif (open_position['type'] == 'BUY' and current_bar['bearish_flip'] == 1) or (open_position['type'] == 'SELL' and current_bar['bullish_flip'] == 1) and dynamic_profit_enabled:
                    exit_reason = "Flip Signal"
                
                if exit_reason:
                    gross_pnl = (exit_price - open_position['entry_price']) * (open_position['lot_size'] / 0.01)
                    closing_commission = calculate_commission(open_position['lot_size'], current_bar.name.hour)
                    net_profit = gross_pnl - closing_commission - open_position['opening_commission']
                    
                    initial_balance += gross_pnl - closing_commission
                    
                    trade_record = {
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_bar.name,
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'lot_size': abs(open_position['lot_size']),
                        'trade_type': open_position['type'],
                        'net_profit': net_profit,
                        'exit_reason': exit_reason,
                        'regime': open_position['regime']
                    }
                    trades.append(trade_record)
                    
                    logger.info(f"🔻 CLOSING {open_position['type']} @ {exit_price:.2f}. Reason: {exit_reason}. Net P/L: ${net_profit:,.2f}")
                    open_position = None
            
            active_specialist_id = current_bar['ce_direction']
            active_specialist_info = squad['specialists'].get(active_specialist_id)
            action_value = 0.0

            if active_specialist_info:
                model, scaler = active_specialist_info['model'], active_specialist_info['scaler']
                expected_features = scaler.feature_names_in_
                obs_df = backtest_df.iloc[i - sequence_length + 1 : i + 1]
                obs_features_subset = obs_df[[col for col in expected_features if col in obs_df.columns]]
                obs_features_aligned = obs_features_subset.reindex(columns=expected_features, fill_value=0)
                obs_features_scaled = scaler.transform(obs_features_aligned)
                obs_space_shape = model.policy.observation_space.shape
                num_placeholders = obs_space_shape[1] - obs_features_scaled.shape[1]
                placeholders = np.zeros((sequence_length, num_placeholders))
                position_type = 1 if open_position and open_position['type'] == 'BUY' else (-1 if open_position and open_position['type'] == 'SELL' else 0)
                placeholders[:, 0] = position_type
                observation_raw = np.concatenate([obs_features_scaled, placeholders], axis=1)
                observation = np.expand_dims(observation_raw, axis=0).astype(np.float32)
                action, _ = model.predict(observation, deterministic=True)
                action_value = action[0][0]
            
            if not open_position:
                current_hour = current_bar.name.hour
                is_trading_hours = not use_time_filter or (trading_start_hour <= current_hour < trading_end_hour)
                if is_trading_hours:
                    total_profit = initial_balance - starting_balance
                    profit_increments = max(0, math.floor(total_profit / profit_threshold_for_step))
                    lot_size = base_lot_size + (profit_increments * lot_step_per_profit)
                    lot_size = round(max(min_lot_size, min(lot_size, max_lot_size)), 2)
                    
                    stop_loss_distance = current_bar['atr'] * atr_multiplier
                    
                    if action_value > 0.5 and (not use_ema_filter or current_bar['ema_fast'] > current_bar['ema_slow']):
                        opening_commission = calculate_commission(lot_size, current_hour)
                        initial_balance -= opening_commission
                        sl, tp = current_bar['close'] - stop_loss_distance, current_bar['close'] + stop_loss_distance
                        open_position = {'type': 'BUY', 'lot_size': lot_size, 'entry_price': current_bar['close'], 
                                         'balance_at_entry': initial_balance, 'stop_loss': sl, 'take_profit': tp,
                                         'entry_time': current_bar.name, 'regime': current_bar['ce_direction'],
                                         'opening_commission': opening_commission}
                        logger.info(f"🔼 ENTERING BUY ({lot_size:.2f} lots) @ {current_bar['close']:.2f} | SL: {sl:.2f}")
                    elif action_value < -0.5 and (not use_ema_filter or current_bar['ema_fast'] < current_bar['ema_slow']):
                        opening_commission = calculate_commission(lot_size, current_hour)
                        initial_balance -= opening_commission
                        sl, tp = current_bar['close'] + stop_loss_distance, current_bar['close'] - stop_loss_distance
                        open_position = {'type': 'SELL', 'lot_size': -lot_size, 'entry_price': current_bar['close'], 
                                         'balance_at_entry': initial_balance, 'stop_loss': sl, 'take_profit': tp,
                                         'entry_time': current_bar.name, 'regime': current_bar['ce_direction'],
                                         'opening_commission': opening_commission}
                        logger.info(f"🔽 ENTERING SELL ({lot_size:.2f} lots) @ {current_bar['close']:.2f} | SL: {sl:.2f}")

            equity_curve.append(equity)
            
            if i < len(backtest_df) -1:
                current_date, previous_date = current_bar.name, previous_bar.name
                if current_date.dayofweek < previous_date.dayofweek:
                    weekly_return = (equity / last_weekly_equity - 1) * 100 if last_weekly_equity > 0 else 0
                    logger.info(f"--- 📅 Weekly Report [Week of {previous_date.strftime('%Y-%m-%d')}] ---")
                    logger.info(f"      End of Week Equity: ${equity:,.2f} | Weekly P/L: {weekly_return:+.2f}%")
                    last_weekly_equity = equity
                if current_date.month != previous_date.month:
                    monthly_return = (equity / last_monthly_equity - 1) * 100 if last_monthly_equity > 0 else 0
                    logger.info(f"--- 🗓️ Monthly Report [{previous_date.strftime('%Y-%B')}] ---")
                    logger.info(f"      End of Month Equity: ${equity:,.2f} | Monthly P/L: {monthly_return:+.2f}%")
                    last_monthly_equity = equity

        logger.info("Backtest simulation complete. Saving results...")
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(paths.RESULTS_DIR / f"hierarchical_backtest_trades{results_suffix}.csv", index=False)
        pd.Series(equity_curve).to_csv(paths.RESULTS_DIR / f"hierarchical_backtest_equity{results_suffix}.csv", index=False, header=['equity'])
        
        if len(equity_curve) > 1:
            performance = calculate_performance_metrics(pd.Series(equity_curve))
            logger.info("\n--- HIERARCHICAL BACKTEST PERFORMANCE ---")
            logger.info(performance)
        else: 
            logger.warning("No trades were executed during the backtest.")

    except Exception as e:
        logger.error(f"A critical error occurred during backtesting: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_hierarchical_backtest()
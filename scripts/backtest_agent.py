# scripts/backtest_agent.py (UPGRADED WITH PROFESSIONAL TRADE MANAGEMENT)

import pandas as pd
import joblib
import os
import sys
import numpy as np

# --- Add project path to allow imports from other directories ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC
from config import paths
from config.init import Config
from models.custom_policy import CustomActorCriticPolicy
from utils.data_transformation import DataTransformer
from utils.logger import setup_logging, get_logger
from utils.metrics import calculate_performance_metrics
from utils.accelerator import DEVICE

def run_backtest():
    """
    Runs a professional backtest with advanced, configurable trade management:
    - Always uses an ATR-based Stop Loss.
    - Can switch between a fixed ATR-based Take Profit and a dynamic Chandelier Exit Take Profit.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING BACKTEST WITH ADVANCED TRADE MANAGEMENT")
    logger.info("="*80)

    try:
        # --- 1. Load Model & Artifacts (Same as before) ---
        logger.info("Loading production agent and artifacts...")
        agent_config = Config()
        policy_kwargs = dict(
            features_extractor_class=CustomActorCriticPolicy,
            features_extractor_kwargs=dict(features_dim=agent_config.get('model.features_dim')),
            use_sde=False
        )
        agent_model = SAC.load(str(paths.FINAL_MODEL_PATH), device=DEVICE, buffer_size=0, policy_kwargs=policy_kwargs)
        agent_scaler = joblib.load(str(paths.FINAL_SCALER_PATH))
        regime_model = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        regime_scaler = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        logger.info("All production artifacts loaded successfully.")

        # --- 2. Process Unseen Data (Same as before) ---
        logger.info(f"Loading and processing unseen data from: {paths.BACKTEST_DATA_FILE}")
        transformer = DataTransformer()
        backtest_df = transformer.load_and_preprocess_data(file_path=str(paths.BACKTEST_DATA_FILE), timeframe='15min')
        
        # Enrich with regime probabilities
        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        X_raw_regime = backtest_df[regime_features].ffill().bfill()
        X_scaled_regime = regime_scaler.transform(X_raw_regime)
        regime_probabilities = regime_model.predict_proba(X_scaled_regime)
        prob_cols = [f'regime_prob_{i}' for i in range(regime_model.n_components)]
        prob_df = pd.DataFrame(regime_probabilities, index=backtest_df.index, columns=prob_cols)
        enriched_df = backtest_df.join(prob_df)

        # Scale final feature set
        feature_cols = [col for col in enriched_df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        scaled_features = agent_scaler.transform(enriched_df[feature_cols])
        
        sim_df = pd.DataFrame(scaled_features, columns=feature_cols, index=enriched_df.index)
        sim_df = sim_df.join(enriched_df[['open', 'high', 'low', 'close']])
        sim_df.reset_index(inplace=True)
        logger.info(f"Prepared {len(sim_df)} data points for backtesting.")


        # --- 3. Simulation with NEW Advanced Trade Management ---
        # --- Dynamic Lot Sizing Configuration ---
        BASE_LOT_SIZE = 0.01
        MAX_LOT_SIZE = 1.7
        PROFIT_STEP_FOR_INCREASE = 100.0 # Increase lot size for every $100 of profit

        def calculate_dynamic_lot_size(equity, initial_balance):
            """Calculates the lot size based on the current profit."""
            profit = equity - initial_balance
            
            # If there is no profit, use the base lot size
            if profit <= 0:
                return BASE_LOT_SIZE
            
            # Calculate how many $100 increments of profit we have
            profit_increments = int(profit // PROFIT_STEP_FOR_INCREASE)
            
            # Each increment adds 0.01 to the lot size
            lot_increase = profit_increments * 0.01
            
            # Calculate the new lot size
            new_lot_size = BASE_LOT_SIZE + lot_increase
            
            # Ensure the lot size does not exceed the maximum allowed
            final_lot_size = min(new_lot_size, MAX_LOT_SIZE)
            
            # Return a valid lot size rounded to 2 decimal places
            return round(final_lot_size, 2)

        # --- Trade Management Configuration (remains the same) ---
        USE_DYNAMIC_TP = False
        ATR_TP_MUL = 3.0
        ATR_SL_MUL = 4.0
        COMMISSION_PER_STANDARD_LOT = 7.0

        # --- Simulation State Initialization ---
        initial_balance = 10000
        equity = initial_balance
        equity_curve = [initial_balance]
        trades = []
        open_position = None

        sequence_length = agent_config.get('environment.sequence_length', 150)
        BUY_THRESHOLD, SELL_THRESHOLD = 0.5, -0.5

        logger.info("Starting simulation loop with DYNAMIC LOT SIZING logic...")
        for i in range(sequence_length, len(sim_df)):
            current_bar = sim_df.iloc[i]
            
            # --- STEP 1: ALWAYS get the model's latest signal on every bar ---
            obs_df = sim_df.iloc[i - sequence_length : i]
            obs_features = obs_df[feature_cols].values
            
            # The position feature should reflect the state *before* this bar's decision
            current_pos_size = 1.0 if open_position else 0.0
            position_feature = np.full((sequence_length, 1), current_pos_size, dtype=np.float32)
            
            observation = np.expand_dims(np.concatenate([obs_features, position_feature], axis=1), axis=0)
            action_continuous, _ = agent_model.predict(observation, deterministic=True)
            action_value = action_continuous[0][0]
            logger.info(f"Bar {i}: Action Value = {action_value}")
            # --- STEP 2: MANAGE any open position using the latest market data and model signal ---
            if open_position:
                close_price = 0
                exit_reason = None

                # a. Check for Stop Loss Hit
                if open_position['type'] == 'BUY' and current_bar['low'] <= open_position['stop_loss']:
                    close_price = open_position['stop_loss']
                    exit_reason = 'Stop Loss Hit'
                elif open_position['type'] == 'SELL' and current_bar['high'] >= open_position['stop_loss']:
                    close_price = open_position['stop_loss']
                    exit_reason = 'Stop Loss Hit'

                # b. Check for Take Profit Hit (if SL not hit)
                if not exit_reason:
                    if USE_DYNAMIC_TP:
                        if (open_position['type'] == 'BUY' and current_bar['bearish_flip'] == 1) or \
                        (open_position['type'] == 'SELL' and current_bar['bullish_flip'] == 1):
                            close_price = current_bar['open']
                            exit_reason = 'Dynamic TP (CE Flip)'
                    else: # Fixed ATR Take Profit
                        if (open_position['type'] == 'BUY' and current_bar['high'] >= open_position['take_profit']):
                            close_price = open_position['take_profit']
                            exit_reason = 'Fixed TP Hit'
                        elif (open_position['type'] == 'SELL' and current_bar['low'] <= open_position['take_profit']):
                            close_price = open_position['take_profit']
                            exit_reason = 'Fixed TP Hit'
                
                # c. Check for Model Counter-Signal (if SL/TP not hit)
                if not exit_reason:
                    if (open_position['type'] == 'BUY' and action_value < SELL_THRESHOLD) or \
                    (open_position['type'] == 'SELL' and action_value > BUY_THRESHOLD):
                        close_price = current_bar['open']
                        exit_reason = 'Model Counter-Signal'

                # If any exit condition was met, process the closed trade
                if exit_reason:
                    # **MODIFICATION: Use the lot size stored with the trade**
                    trade_lot_size = open_position['lot_size']
                    
                    price_difference = close_price - open_position['entry_price']
                    gross_pnl = price_difference * (trade_lot_size / 0.01)
                    if open_position['type'] == 'SELL':
                        gross_pnl = -gross_pnl

                    commission = trade_lot_size * COMMISSION_PER_STANDARD_LOT
                    net_pnl = gross_pnl - commission

                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_bar['timestamp'],
                        'lot_size': trade_lot_size, # Log the lot size for analysis
                        'type': open_position['type'],
                        'entry_price': open_position['entry_price'],
                        'exit_price': close_price,
                        'net_profit': net_pnl,
                        'exit_reason': exit_reason
                    })
                    equity += net_pnl
                    open_position = None
                    
            
            if not open_position:
                # --- STEP 3: LOOK FOR NEW ENTRIES (only if no position is currently open) ---
                current_lot_size = calculate_dynamic_lot_size(equity, initial_balance)

                if action_value > BUY_THRESHOLD:
                    entry_price = current_bar['open']
                    atr_at_entry = current_bar['atr']
                    stop_loss = entry_price - (atr_at_entry * ATR_SL_MUL)
                    take_profit = entry_price + (atr_at_entry * ATR_TP_MUL)
                    logger.info(f"Action value: {action_value}, Entry Price: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
                    # **Store the calculated lot size with the position**
                    open_position = {'type': 'BUY', 'lot_size': current_lot_size, 'entry_price': entry_price, 'entry_time': current_bar['timestamp'], 'stop_loss': stop_loss, 'take_profit': take_profit}
                
                elif action_value < SELL_THRESHOLD:
                    entry_price = current_bar['open']
                    atr_at_entry = current_bar['atr']
                    stop_loss = entry_price + (atr_at_entry * ATR_SL_MUL)
                    take_profit = entry_price - (atr_at_entry * ATR_TP_MUL)
                    logger.info(f"Action value: {action_value}, Entry Price: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
                    # **Store the calculated lot size with the position**
                    open_position = {'type': 'SELL', 'lot_size': current_lot_size, 'entry_price': entry_price, 'entry_time': current_bar['timestamp'], 'stop_loss': stop_loss, 'take_profit': take_profit}

            # --- STEP 4: UPDATE EQUITY CURVE ---
            temp_equity = equity
            if open_position:
                # **MODIFICATION: Use the trade's specific lot size for unrealized PnL**
                trade_lot_size = open_position['lot_size']
                price_difference = current_bar['close'] - open_position['entry_price']
                unrealized_pnl = price_difference * (trade_lot_size / 0.01)
                if open_position['type'] == 'SELL':
                    unrealized_pnl = -unrealized_pnl
                temp_equity += unrealized_pnl
            equity_curve.append(temp_equity)

        # --- 4. Save and Display Final Results ---
        trades_df = pd.DataFrame(trades)
        equity_series = pd.Series(equity_curve)

        trades_df.to_csv(paths.RESULTS_DIR / "final_backtest_trades.csv", index=False)
        equity_series.to_csv(paths.RESULTS_DIR / "final_backtest_equity.csv", index=False)
        # --- ADD THIS LINE ---
        sim_df.to_csv(paths.RESULTS_DIR / "final_backtest_simulation_data.csv", index=False)
        # ---------------------
        logger.info(f"Backtest complete. Results saved to {paths.RESULTS_DIR}")

        # Final Report
        performance_stats = calculate_performance_metrics(equity_series, periods_per_year=252*24*4)
        print("\n" + "="*50)
        print("      BACKTEST REPORT (WITH TRADE MANAGEMENT)")
        print("="*50)
        print(f"TP Strategy:             {'Dynamic (CE Flip)' if USE_DYNAMIC_TP else f'Fixed (ATR x{ATR_TP_MUL})'}")
        print(f"SL Strategy:             Fixed (ATR x{ATR_SL_MUL})")
        print(f"Total Trades:            {len(trades_df)}")
        if not trades_df.empty:
            win_rate = (trades_df['net_profit'] > 0).mean()
            losing_trades = trades_df[trades_df['net_profit'] < 0]['net_profit']
            profit_factor = trades_df[trades_df['net_profit'] > 0]['net_profit'].sum() / abs(losing_trades.sum()) if not losing_trades.empty else np.inf
            print(f"Win Rate:                {win_rate * 100:.2f}%")
            print(f"Profit Factor:           {profit_factor:.2f}")
        
        print("\nPORTFOLIO METRICS:")
        print(f"Initial Portfolio Value: ${equity_series.iloc[0]:,.2f}")
        print(f"Final Portfolio Value:   ${equity_series.iloc[-1]:,.2f}")
        print(f"Total Net Return:        {((equity_series.iloc[-1] / equity_series.iloc[0]) - 1) * 100:.2f}%")
        print(f"Max Drawdown:            {performance_stats['max_drawdown'] * 100:.2f}%")
        print(f"Sharpe Ratio:            {performance_stats['sharpe_ratio']:.2f}")
        print("="*50)

    except Exception as e:
        logger.error(f"An error occurred during backtesting: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_backtest()
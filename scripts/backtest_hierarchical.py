# scripts/backtest_hierarchical.py
# FINAL, FEATURE-ALIGNED, AND EXCEPTION-HANDLED BACKTESTING SIMULATOR

import pandas as pd
import joblib
import os
import sys
import numpy as np
import warnings
# --- Add project root to path for robust imports ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from stable_baselines3 import SAC
from config import paths
from config.init import Config
from utils.logger import setup_logging, get_logger
from utils.metrics import calculate_performance_metrics
from utils.accelerator import DEVICE
from utils.data_transformation import DataTransformer

def run_hierarchical_backtest(results_suffix: str = ""):
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"INITIATING TRUE-TO-LIVE HIERARCHICAL BACKTEST...")

    try:
        # --- إعدادات الفلاتر والصفقات ---
        use_dynamic_tp = True
        atr_multiplier_sl = 2.0
        atr_multiplier_tp = 4.0
        
        # --- فلتر EMA ---
        use_ema_filter = True
        ema_period = 50
        
        logger.info(f"Backtest Configuration: Dynamic TP={use_dynamic_tp}, SL ATR={atr_multiplier_sl}, TP ATR={atr_multiplier_tp}, EMA Filter={use_ema_filter}")
        
        agent_config = Config()
        squad = {}
        model_dir = paths.FINAL_MODEL_DIR
        logger.info(f"Loading Command Unit and Specialist Squad from: {model_dir}")
        
        squad['classifier'] = joblib.load(model_dir / "regime_gmm_model.joblib")
        squad['regime_scaler'] = joblib.load(model_dir / "regime_robust_scaler.joblib")
        num_regimes = squad['classifier'].n_components
        
        squad['specialists'] = {}
        for i in range(num_regimes):
            specialist_dir = model_dir / f"specialist_regime_{i}"
            model_path = specialist_dir / "models/best_model.zip"
            scaler_path = specialist_dir / f"specialist_scaler_regime_{i}.joblib"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                squad['specialists'][i] = {
                    'model': SAC.load(model_path, device=DEVICE),
                    'scaler': joblib.load(scaler_path)
                }
                logger.info(f"-> Specialist for Regime {i} loaded successfully.")
        
        if not squad['specialists']:
            logger.error("No specialist models were loaded. Cannot proceed.")
            return

        logger.info("STEP 1: Loading and processing completely unseen test data...")
        transformer = DataTransformer()
        backtest_df = transformer.load_and_preprocess_data(file_path=str(paths.BACKTEST_DATA_FILE), timeframe="15min")
        
        # --- حساب وإضافة فلتر EMA إلى البيانات ---
        if use_ema_filter:
            backtest_df[f'ema_{ema_period}'] = backtest_df['close'].ewm(span=ema_period, adjust=False).mean()
            logger.info(f"Calculated and added EMA({ema_period}) filter column.")

        if 'timestamp' not in backtest_df.columns:
            backtest_df.reset_index(inplace=True)
        if 'atr' not in backtest_df.columns:
            logger.error("ATR column is missing from the data. Cannot calculate SL/TP.")
            return
        logger.info(f"Test data processed. Shape: {backtest_df.shape}")

        initial_balance = agent_config.get('environment.initial_balance')
        equity = initial_balance
        equity_curve = [initial_balance]
        trades = []
        open_position = None
        sequence_length = agent_config.get('environment.sequence_length', 30)
        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        
        logger.info("STEP 2: Starting true-to-live simulation loop (candle by candle)...")
        for i in range(sequence_length, len(backtest_df)):
            current_bar = backtest_df.iloc[i]
            
            current_regime_features_raw = current_bar[regime_features].values.reshape(1, -1)
            current_regime_features_scaled = squad['regime_scaler'].transform(current_regime_features_raw)
            current_regime = squad['classifier'].predict(current_regime_features_scaled)[0]
            
            # if current_regime in [2, 4, 7, 8, 9]:
            #     continue
            
            active_specialist_info = squad['specialists'].get(current_regime)
            action_value = 0.0

            if active_specialist_info:
                model = active_specialist_info['model']
                scaler = active_specialist_info['scaler']
                
                # --- هنا تكمن الحماية ---
                # 1. استدعاء قائمة الميزات التي يعرفها النموذج فقط
                expected_features = scaler.feature_names_in_
                
                # 2. أخذ نافذة البيانات الحالية (مع وجود عمود EMA فيها)
                obs_df = backtest_df.iloc[i - sequence_length + 1 : i + 1]
                
                # 3. اختيار الميزات المتوقعة فقط وتجاهل الباقي (مثل EMA)
                current_features = [col for col in expected_features if col in obs_df.columns]
                obs_features_subset = obs_df[current_features]

                # 4. إعادة ترتيب الأعمدة وضمان تطابقها 100% مع بيانات التدريب
                obs_features_aligned = obs_features_subset.reindex(columns=expected_features, fill_value=0)
                obs_features_scaled = scaler.transform(obs_features_aligned)

                # --- (باقي الكود يستخدم البيانات الآمنة فقط) ---
                obs_space_shape = model.policy.observation_space.shape
                features_dim = obs_space_shape[1]
                num_placeholders = features_dim - obs_features_scaled.shape[1]
                
                placeholders = np.zeros((sequence_length, num_placeholders))
                placeholders[:, 0] = 1.0 if open_position else 0.0 

                observation_raw = np.concatenate([obs_features_scaled, placeholders], axis=1)
                observation = np.expand_dims(observation_raw, axis=0).astype(np.float32)
                action, _ = model.predict(observation, deterministic=True)
                action_value = action[0][0]
            
            if open_position:
                # (منطق إغلاق الصفقة يبقى كما هو)
                close_trade = False; exit_price = None; status = ''
                trade_type = open_position['type']; sl_price = open_position['stop_loss']; tp_price = open_position['take_profit']
                current_high = current_bar['high']; current_low = current_bar['low']
                if trade_type == 'BUY' and current_low <= sl_price: close_trade, exit_price, status = True, sl_price, 'Stop Loss'
                elif trade_type == 'SELL' and current_high >= sl_price: close_trade, exit_price, status = True, sl_price, 'Stop Loss'
                if not close_trade and not use_dynamic_tp:
                    if trade_type == 'BUY' and current_high >= tp_price: close_trade, exit_price, status = True, tp_price, 'Take Profit'
                    elif trade_type == 'SELL' and current_low <= tp_price: close_trade, exit_price, status = True, tp_price, 'Take Profit'
                if not close_trade:
                    if (trade_type == 'BUY' and action_value < 0.1) or (trade_type == 'SELL' and action_value > -0.1):
                        close_trade, exit_price, status = True, current_bar['open'], 'Signal'
                if close_trade:
                    pnl_multiplier = 1 if open_position['type'] == 'BUY' else -1
                    pnl = (exit_price - open_position['entry_price']) * pnl_multiplier
                    equity += pnl
                    trades.append({'entry_time': open_position['entry_time'], 'exit_time': current_bar['timestamp'], 'type': open_position['type'], 'entry_price': open_position['entry_price'], 'exit_price': exit_price, 'net_profit': pnl, 'regime': open_position['regime_at_entry'], 'status': status, 'stop_loss': sl_price, 'take_profit': tp_price, 'balance_after_trade': equity})
                    open_position = None
            
            if not open_position:
                BUY_THRESHOLD, SELL_THRESHOLD = 0.1, -0.0001
                
                # --- استخدام EMA للفلترة فقط ---
                ema_allows_buy = True
                ema_allows_sell = True
                if use_ema_filter:
                    current_ema = current_bar[f'ema_{ema_period}']
                    if current_bar['open'] <= current_ema: ema_allows_buy = False
                    if current_bar['open'] >= current_ema: ema_allows_sell = False

                if action_value > BUY_THRESHOLD and ema_allows_buy: entry_type = 'BUY'
                elif action_value < SELL_THRESHOLD and ema_allows_sell: entry_type = 'SELL'
                else: entry_type = None

                if entry_type:
                    atr_at_entry = current_bar['atr']; sl_distance = atr_at_entry * atr_multiplier_sl; tp_distance = atr_at_entry * atr_multiplier_tp; entry_price = current_bar['open']
                    if entry_type == 'BUY': 
                        sl = entry_price - sl_distance; 
                        if use_dynamic_tp:
                            tp = 0.0
                        else:
                            tp = entry_price + tp_distance
                    else: 
                        sl = entry_price + sl_distance
                        if use_dynamic_tp:
                            tp = 0.0
                        else:
                            tp = entry_price - tp_distance
                    open_position = {'type': entry_type, 'entry_price': entry_price, 'entry_time': current_bar['timestamp'], 'regime_at_entry': current_regime, 'stop_loss': sl, 'take_profit': tp}

            current_equity = equity
            if open_position:
                pnl_multiplier = 1 if open_position['type'] == 'BUY' else -1
                current_equity += (current_bar['close'] - open_position['entry_price']) * pnl_multiplier
            equity_curve.append(current_equity)

        logger.info("Hierarchical backtest simulation complete. Saving results...")
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(paths.RESULTS_DIR / f"hierarchical_backtest_trades{results_suffix}.csv", index=False, encoding='utf-8-sig')
        pd.Series(equity_curve).to_csv(paths.RESULTS_DIR / f"hierarchical_backtest_equity{results_suffix}.csv", index=False, header=['equity'])
        
        if not trades_df.empty:
            performance = calculate_performance_metrics(pd.Series(equity_curve))
            logger.info("\n--- HIERARCHICAL BACKTEST PERFORMANCE ---")
            logger.info(performance)
        else:
            logger.warning("No trades were executed during the backtest.")

    except Exception as e:
        logger.error(f"A critical error occurred during hierarchical backtesting: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_hierarchical_backtest()
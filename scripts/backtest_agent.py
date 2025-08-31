# scripts/backtest_agent.py (FULLY UPGRADED FOR PROFESSIONAL VALIDATION)

import pandas as pd
import joblib
import os
import sys

# --- Add project path to allow imports from other directories ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC
from config import paths
from envs.trading_env import TradingEnv
from utils.data_transformation import DataTransformer
from utils.logger import setup_logging, get_logger
from utils.metrics import calculate_performance_metrics

def run_backtest():
    """
    Runs a professional backtest by applying the FULL data enrichment pipeline
    to unseen data before simulating the final agent's performance. This provides
    a robust, out-of-sample validation of the entire trading system.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING PROFESSIONAL BACKTESTING PROCESS")
    logger.info("="*80)
    
    try:
        # --- 1. Load all necessary production artifacts ---
        logger.info("Loading production agent, scalers, and regime model...")
        
        # Load the final trained trading agent
        agent_model = SAC.load(str(paths.FINAL_MODEL_PATH))
        
        # Load the scaler that was used for the agent's training data
        agent_scaler = joblib.load(str(paths.FINAL_SCALER_PATH))
        
        # Load the unsupervised model for market regime detection
        regime_model = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        
        # Load the scaler that was used for the regime model's training data
        regime_scaler = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        
        logger.info("All production artifacts loaded successfully.")

        # --- 2. Load and apply initial feature engineering to UNSEEN backtest data ---
        logger.info(f"Loading and processing unseen backtest data from: {paths.BACKTEST_DATA_FILE}")
        transformer = DataTransformer()
        backtest_df = transformer.load_and_preprocess_data(file_path=str(paths.BACKTEST_DATA_FILE))

        # --- 3. Enrich the UNSEEN data with market regime probabilities on the fly ---
        # This step simulates what a live agent would do with new data.
        logger.info("Enriching unseen data with market regime vectors...")
        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        
        # Ensure data is clean before prediction
        X_raw = backtest_df[regime_features].ffill().bfill()
        X_scaled = regime_scaler.transform(X_raw)
        
        # Get the probability vectors from the GMM
        regime_probabilities = regime_model.predict_proba(X_scaled)
        prob_cols = [f'regime_prob_{i}' for i in range(regime_model.n_components)]
        prob_df = pd.DataFrame(regime_probabilities, index=backtest_df.index, columns=prob_cols)
        
        # Merge the new regime features into the backtest dataframe
        enriched_backtest_df = backtest_df.join(prob_df)

        # --- 4. Scale the final enriched features with the AGENT's specific scaler ---
        logger.info("Scaling final feature set for the agent...")
        feature_cols = [col for col in enriched_backtest_df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        scaled_features = agent_scaler.transform(enriched_backtest_df[feature_cols])
        
        final_backtest_df = pd.DataFrame(scaled_features, columns=feature_cols, index=enriched_backtest_df.index)
        final_backtest_df['close'] = enriched_backtest_df['close']
        
        logger.info(f"Prepared {len(final_backtest_df)} data points for backtesting.")

        # --- 5. Run the simulation in the professional environment ---
        backtest_env = TradingEnv(final_backtest_df)
        obs, _ = backtest_env.reset()
        done = False
        equity_curve = [backtest_env.initial_balance]
        
        logger.info("Starting simulation loop...")
        while not done:
            # Get deterministic action from the trained agent
            action, _ = agent_model.predict(obs, deterministic=True)
            obs, _, done, _, info = backtest_env.step(action)
            equity_curve.append(info['equity'])
        logger.info("Simulation finished.")

        # --- 6. Calculate and display professional performance metrics ---
        equity_series = pd.Series(equity_curve)
        performance_stats = calculate_performance_metrics(equity_series)

        # Save the results for further analysis or plotting
        equity_series.to_csv(paths.RESULTS_DIR / "final_backtest_equity.csv", index=False)
        logger.info(f"Backtest equity curve saved to: {paths.RESULTS_DIR / 'final_backtest_equity.csv'}")

        # Print the final, comprehensive report
        print("\n" + "="*50)
        print("      PROFESSIONAL BACKTESTING REPORT")
        print("="*50)
        print(f"Initial Portfolio Value: ${equity_series.iloc[0]:,.2f}")
        print(f"Final Portfolio Value:   ${equity_series.iloc[-1]:,.2f}")
        print(f"Total Return:            {((equity_series.iloc[-1] / equity_series.iloc[0]) - 1) * 100:.2f}%")
        print("-"*50)
        print("RISK-ADJUSTED METRICS:")
        print(f"Annualized Sharpe Ratio: {performance_stats['sharpe_ratio']:.2f}")
        print(f"Annualized Sortino Ratio: {performance_stats['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:            {performance_stats['calmar_ratio']:.2f}")
        print(f"Maximum Drawdown:        {performance_stats['max_drawdown'] * 100:.2f}%")
        print("="*50)

    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
        logger.error("Please ensure the full training pipeline has been run successfully before backtesting.")
    except Exception as e:
        logger.error(f"An error occurred during backtesting: {e}", exc_info=True)

if __name__ == "__main__":
    run_backtest()
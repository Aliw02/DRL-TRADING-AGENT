import argparse
from scripts.train_agent import train_agent
from scripts.backtest_agent import backtest_agent
from scripts.plot_results import plot_results
from utils.logger import setup_logging, get_logger

def main():
    """Main entry point for the DRL trading agent"""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="XAUUSD DRL Trading Agent")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--backtest", action="store_true", help="Backtest the agent")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--model-path", type=str, default="results/models/xauusd_trading_agent_final", 
                       help="Path to the trained model")
    
    args = parser.parse_args()
    
    try:
        # Train if requested
        if args.train:
            logger.info("Starting training process")
            model = train_agent()
        else:
            model = None
        
        # Backtest if requested
        if args.backtest:
            logger.info("Starting backtesting process")
            equity_curve, trades, metrics = backtest_agent(args.model_path)
            
            # Print metrics
            print("\nBacktest Results:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        
        # Plot if requested
        if args.plot and args.backtest:
            logger.info("Generating plots")
            plot_results(equity_curve, trades)
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
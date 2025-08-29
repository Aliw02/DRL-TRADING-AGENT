# XAUUSD DRL Trading Agent

A professional Deep Reinforcement Learning trading agent for XAUUSD (Gold vs. US Dollar) on the M5 timeframe, learning from Chandelier Exit signals.

## Project Structure

## Project Structure
xauusd_drl_trading/
├── config/ # Configuration files
├── data/ # Data files
├── envs/ # Trading environment
├── models/ # Custom policy networks
├── utils/ # Utility functions
├── scripts/ # Training and evaluation scripts
├── tests/ # Unit tests
├── results/ # Outputs (models, logs, plots)
├── main.py # Main entry point
├── requirements.txt # Dependencies
└── README.md # This file


## Key Features

- Dual-stream neural network architecture with specialized CE signal processing
- Agent learns from CE signals rather than being forced to follow them
- Professional-grade error handling and logging
- Comprehensive performance metrics and visualization
- Modular design for easy extension and maintenance

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your XAUUSD data in data/XAUUSD-FULL.csv

3. Run the agent:
   ```bash
   python main.py --train --backtest --plot
   ```

   Approach
The agent uses Chandelier Exit signals as input features but learns autonomously when to follow or ignore them based on profitability. This allows for more nuanced trading strategies that can adapt to changing market conditions.

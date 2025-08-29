#!/usr/bin/env python3
"""
Project Initialization Script for XAUUSD DRL Trading Agent
Creates the complete folder structure and necessary files
"""

import os
import stat
from pathlib import Path

def create_file(path, content):
    """Create a file with the given content"""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    # Ensure files are written using UTF-8 to avoid encoding errors on Windows
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    print(f"Created: {path}")

def initialize_project():
    """Initialize the complete project structure"""
    
    print("Creating XAUUSD DRL Trading Agent project structure...")
    
    # Create directory structure
    directories = [
        "config",
        "data",
        "envs",
        "models", 
        "utils",
        "scripts",
        "tests",
        "results/models",
        "results/logs",
        "results/plots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    for pkg in ["config", "data", "envs", "models", "utils", "scripts", "tests"]:
        create_file(f"{pkg}/__init__.py", "# Package initialization\n")
    
    # Create configuration files
    config_content = """# Training parameters
training:
  total_timesteps: 100000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01

# Environment parameters
environment:
  initial_balance: 10000
  sequence_length: 150
  action_penalty: -0.01  # Small penalty for trading actions

# Data parameters
data:
  file_path: "data/XAUUSD-FULL.csv"
  train_test_split: 0.8
  feature_columns:
    - direction
    - bullish_flip
    - bearish_flip
    - dist_to_stop
    - adx
    - rsi
    - plus_di
    - minus_di
    - ma_fast
    - ma_slow
    - macd
    - macd_signal
    - bb_width
    - roc
    - volatility
    - volume_ratio

# Model parameters
model:
  features_dim: 256
  ce_stream_hidden: 64
  context_stream_hidden: 128
  position_stream_hidden: 16

# Logging parameters
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "results/logs/trading_agent.log"
"""
    create_file("config/config.yaml", config_content)
    
    # Create logger configuration
    logging_config_content = """version: 1
formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: results/logs/trading_agent.log
    maxBytes: 10485760
    backupCount: 5
loggers:
  __main__:
    level: DEBUG
    handlers: [console, file]
    propagate: no
root:
  level: DEBUG
  handlers: [console, file]
"""
    create_file("config/logging.yaml", logging_config_content)
    
    # Create requirements.txt
    requirements_content = """stable-baselines3==2.0.0
gymnasium==0.29.1
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.3.0
torch==2.0.1
pyyaml==6.0
tensorboard==2.13.0
numba==0.57.1
"""
    create_file("./requirements.txt", requirements_content)
    
    # Create .gitignore
    gitignore_content = """# Data
data/*.csv
!data/XAUUSD-FULL.csv

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Results
results/models/
results/logs/
results/plots/
"""
    create_file("./.gitignore", gitignore_content)

    # Create README.md
    readme_content = """# XAUUSD DRL Trading Agent

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
"""
    create_file("./README.md", readme_content)
    
    # Create empty data file placeholder
    create_file("data/XAUUSD-FULL.csv", "timestamp,open,high,low,close,volume\n")

    # Create environment file
    env_content = """import gymnasium as gym
    from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config import config

logger = get_logger(name)

class TradingEnv(gym.Env):
    \"""Trading environment for XAUUSD with Chandelier Exit features\"""\"
    metadata = {'render.modes': ['human']}

def __init__(self, df, initial_balance=None, sequence_length=None):
    super(TradingEnv, self).__init__()
    
    # Load configuration
    self.initial_balance = initial_balance or config.get('environment.initial_balance', 10000)
    self.sequence_length = sequence_length or config.get('environment.sequence_length', 150)
    self.action_penalty = config.get('environment.action_penalty', -0.01)
    
    # Validate input data
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if len(df) < self.sequence_length:
        raise ValueError(f"DataFrame must have at least {self.sequence_length} rows")
    
    self.df = df.reset_index(drop=True)
    self.current_step = self.sequence_length
    self.max_steps = len(self.df) - 1
    
    # Actions: 0=Hold, 1=Buy, 2=Sell
    self.action_space = spaces.Discrete(3)
    
    # Observation space
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, 
        shape=(self.sequence_length, 18),  # 16 features + position + equity
        dtype=np.float32
    )
    
    # Reset environment
    self.reset()
    logger.info("Trading environment initialized successfully")

def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.current_step = self.sequence_length
    self.position = 0  # 0: no position, 1: long
    self.entry_price = 0
    self.equity = self.initial_balance
    self.prev_equity = self.equity
    self.trade_count = 0
    
    logger.debug("Environment reset")
    return self._next_observation(), {}

def step(self, action):
    try:
        # Validate action
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}")
        
        # Execute action
        self._take_action(action)
        self.current_step += 1
        
        # Calculate reward
        reward = self._get_reward(action)
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Update previous equity
        self.prev_equity = self.equity
        
        # Get next observation
        obs = self._next_observation()
        
        # Info dict
        info = {
            'step': self.current_step,
            'equity': self.equity,
            'position': self.position,
            'reward': reward,
            'trade_count': self.trade_count
        }
        
        return obs, reward, done, False, info
        
    except Exception as e:
        logger.error(f"Error in step function: {e}")
        raise

def _take_action(self, action):
    try:
        current_price = self.df.loc[self.current_step, 'close']
        
        # Update equity based on current position
        if self.position == 1:
            self.equity = self.initial_balance + (current_price - self.entry_price)
        
        # Action logic
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.trade_count += 1
                logger.debug(f"Buy at step {self.current_step}, price: {current_price}")
        elif action == 2:  # Sell
            if self.position == 1:
                # Close the long position (realize PnL)
                self.initial_balance = self.equity
                self.position = 0
                self.entry_price = 0
                self.trade_count += 1
                logger.debug(f"Sell at step {self.current_step}, price: {current_price}")
                
    except Exception as e:
        logger.error(f"Error taking action: {e}")
        raise

def _get_reward(self, action):
    try:
        # Primary reward: change in equity
        equity_change = self.equity - self.prev_equity
        
        # Action penalty
        action_penalty = 0
        if action != 0:
            action_penalty = self.action_penalty
        
        reward = equity_change + action_penalty
        return reward
        
    except Exception as e:
        logger.error(f"Error calculating reward: {e}")
        return 0  # Return neutral reward on error

def _next_observation(self):
    try:
        # Get the data from the current step and the previous (sequence_length-1) steps
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step
        
        # Handle edge cases
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(self.df):
            end_idx = len(self.df) - 1
            
        obs_df = self.df.loc[start_idx:end_idx]
        
        # Extract the 16 features
        features = obs_df.values
        
        # Ensure we have the right sequence length
        if len(features) < self.sequence_length:
            # Pad with zeros if needed
            padding = np.zeros((self.sequence_length - len(features), features.shape[1]))
            features = np.vstack([padding, features])
        
        # Create an array for the additional features: position and equity
        additional_features = np.ones((self.sequence_length, 2))
        additional_features[:, 0] = self.position
        additional_features[:, 1] = self.equity
        
        # Combine
        obs = np.concatenate([features, additional_features], axis=1)
        
        return obs.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error creating observation: {e}")
        # Return a safe observation on error
        return np.zeros((self.sequence_length, 18), dtype=np.float32)

def render(self, mode='human'):
    print(f'Step: {self.current_step}, Equity: {self.equity:.2f}, Position: {self.position}')
    """
    create_file("envs/trading_env.py", env_content)
    
    # Create custom policy file
    policy_content = """import torch
    import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.logger import get_logger
from config import config

logger = get_logger(name)

class CustomActorCriticPolicy(BaseFeaturesExtractor):
    \"""Dual-stream policy network with specialized CE signal processing\"""
    
def __init__(self, observation_space, features_dim=None):
    # Load configuration
    features_dim = features_dim or config.get('model.features_dim', 256)
    ce_stream_hidden = config.get('model.ce_stream_hidden', 64)
    context_stream_hidden = config.get('model.context_stream_hidden', 128)
    position_stream_hidden = config.get('model.position_stream_hidden', 16)
    
    super(CustomActorCriticPolicy, self).__init__(observation_space, features_dim)
    
    self.sequence_length = observation_space.shape[0]
    self.feature_dim = observation_space.shape[1]
    
    # CE Stream (first 4 features)
    self.ce_stream = nn.LSTM(
        input_size=4, 
        hidden_size=ce_stream_hidden, 
        num_layers=1,
        batch_first=True,
        dropout=0.1
    )
    
    # Context Stream (next 12 features)
    self.context_stream = nn.LSTM(
        input_size=12, 
        hidden_size=context_stream_hidden, 
        num_layers=1,
        batch_first=True,
        dropout=0.1
    )
    
    # Position and equity stream (last 2 features)
    self.position_stream = nn.Sequential(
        nn.Linear(2, position_stream_hidden),
        nn.ReLU(),
        nn.Dropout(0.1)
    )
    
    # Calculate input size for fully connected layers
    fc_input_size = ce_stream_hidden + context_stream_hidden + position_stream_hidden
    
    # Fully connected layers
    self.fc = nn.Sequential(
        nn.Linear(fc_input_size, features_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(features_dim, features_dim),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
    
    # Value and policy heads
    self.value_net = nn.Sequential(
        nn.Linear(features_dim, 1)
    )
    
    self.policy_net = nn.Sequential(
        nn.Linear(features_dim, 3),
        nn.Softmax(dim=-1)
    )
    
    # Initialize weights
    self._initialize_weights()
    
    logger.info(f"Custom policy network initialized with features_dim: {features_dim}")

def _initialize_weights(self):
    \"\"\"Initialize weights for better training stability\"\"\"
    for name, param in self.named_parameters():
        if 'weight' in name:
            if 'lstm' in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)

def forward(self, observations):
    try:
        # Split the input into streams
        ce_features = observations[:, :, :4]    # CE features
        context_features = observations[:, :, 4:16]  # Context features
        position_features = observations[:, -1, 16:18]  # Position and equity (take only the last step)
        
        # Process CE stream
        ce_out, (ce_hn, _) = self.ce_stream(ce_features)
        ce_out = ce_hn[-1]  # Take the last hidden state
        
        # Process context stream
        context_out, (context_hn, _) = self.context_stream(context_features)
        context_out = context_hn[-1]
        
        # Process position stream
        position_out = self.position_stream(position_features)
        
        # Concatenate all streams
        combined = torch.cat([ce_out, context_out, position_out], dim=1)
        
        # Process through fully connected layers
        features = self.fc(combined)
        
        # Value and policy outputs
        value = self.value_net(features)
        policy_logits = self.policy_net(features)
        
        return features, value, policy_logits
        
    except Exception as e:
        logger.error(f"Error in policy forward pass: {e}")
        # Return safe values on error
        batch_size = observations.shape[0]
        features = torch.zeros(batch_size, self.features_dim, device=observations.device)
        value = torch.zeros(batch_size, 1, device=observations.device)
        policy_logits = torch.ones(batch_size, 3, device=observations.device) / 3
        return features, value, policy_logits
        
        """
    create_file("models/custom_policy.py", policy_content)
    
    
    # Create logger utility
    logger_content = """import logging
    import logging.config
    import yaml
    import os
    from pathlib import Path

    def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO):
        \"\"\"Setup logging configuration\"\"\"
        path = default_path
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
          logging.basicConfig(level=default_level)
            
        return logging.getLogger(__name__)
        
    def get_logger(name):
        \"\"\"Get a logger with the given name\"\"\"
        return logging.getLogger(name)

    # Create logs directory if it doesn't exist
    Path("results/logs").mkdir(parents=True, exist_ok=True)
"""
    create_file("utils/logger.py", logger_content)
    
    # Create configuration loader
    config_loader_content = """import yaml
    import os
from pathlib import Path

class Config:
    \"\"\"Configuration management class\"\"\"
   def __init__(self, config_path="config/config.yaml"):
    self.config_path = config_path
    self.config = self._load_config()

def _load_config(self):
    \"\"\"Load configuration from YAML file\"\"\"
    if not os.path.exists(self.config_path):
        raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    with open(self.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get(self, key, default=None):
    \"\"\"Get configuration value by key\"\"\"
    keys = key.split('.')
    value = self.config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def __getitem__(self, key):
    return self.get(key)
Global configuration instance
config = Config()
"""
    create_file("config/init.py", config_loader_content)
    # Create main.py
    main_content = """import argparse
    from scripts.train_agent import train_agent
from scripts.backtest_agent import backtest_agent
from scripts.plot_results import plot_results
from utils.logger import setup_logging, get_logger

def main():
    \"\"\"Main entry point for the DRL trading agent\"\"\"
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
        print("\\nBacktest Results:")
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
"""
    create_file("main.py", main_content)
    
    
    # Create placeholder files for the remaining components
    placeholder_files = [
        ("utils/custom_indicators.py", "# Technical indicators implementation\n"),
        ("utils/data_transformation.py", "# Data loading and preprocessing\n"),
        ("utils/metrics.py", "# Performance metrics calculation\n"),
        ("scripts/train_agent.py", "# Training script\n"),
        ("scripts/backtest_agent.py", "# Backtesting script\n"),
        ("scripts/plot_results.py", "# Results plotting script\n"),
        ("tests/test_indicators.py", "# Tests for indicators\n"),
        ("tests/test_env.py", "# Tests for trading environment\n")
    ]

    for file_path, content in placeholder_files:
        create_file(file_path, content)

    print("\nProject structure initialized successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Add your XAUUSD M5 data to data/XAUUSD-FULL.csv")
    print("3. Implement the remaining components in the respective files")
    print("4. Run the agent: python main.py --train --backtest --plot")


if __name__ == "__main__":
    initialize_project()

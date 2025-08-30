# 🤖 Advanced Deep Reinforcement Learning Trading Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable Baselines3](https://img.shields.io/badge/SB3-2.0.0-green.svg)](https://stable-baselines3.readthedocs.io/)

A state-of-the-art Deep Reinforcement Learning trading agent implementing a sophisticated **Hybrid CNN-Transformer architecture** for algorithmic trading in financial markets. Specifically optimized for high-frequency data (1-minute timeframe) on assets like XAUUSD (Gold).

## 🌟 Key Features

### 🧠 Advanced Architecture
- **Hybrid CNN-Transformer Model**: Combines CNN for local pattern recognition with Transformer encoders for global context analysis
- **Multi-Objective Reward Function**: Optimizes for profit, risk-adjusted returns (Sortino Ratio), and capital preservation
- **PPO Algorithm**: Implements Proximal Policy Optimization for stable and efficient learning

### 📊 Sophisticated Feature Engineering
- **Multi-Timeframe Analysis**: Integrates M15 and H1 indicators (RSI, ADX) for broader market context
- **Volatility Normalization**: ATR-based normalization for regime-independent signals
- **Candlestick Pattern Recognition**: Automated detection of market psychology patterns
- **Temporal Features**: Hour/day cyclical encoding for session dynamics understanding

### 🔄 Professional Training Methodology
- **Walk-Forward Optimization**: Trains on expanding windows to ensure robustness
- **Data Pipeline**: Efficient two-stage processing for handling massive datasets (10+ years)
- **Memory Optimization**: Parquet format for reduced RAM usage during training

### 📈 Analysis & Interpretability
- **SHAP Integration**: Model interpretability through SHapley Additive exPlanations
- **Feature Importance**: Permutation importance analysis for feature ranking
- **Performance Metrics**: Comprehensive backtesting with professional risk metrics

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Core Language** | Python | 3.10+ |
| **Deep Learning** | PyTorch | 2.0.1 |
| **Reinforcement Learning** | Stable Baselines3 | 2.0.0 |
| **Data Processing** | Pandas, PyArrow | 2.0.3 |
| **Technical Analysis** | TA-Lib | 0.4.26 |
| **Visualization** | Matplotlib | 3.7.2 |
| **Model Analysis** | SHAP, Scikit-learn | 1.3.0 |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- At least 16GB RAM (for large datasets)
- CUDA-compatible GPU (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/Aliw02/DRL-TRADING-AGENT.git
cd DRL-TRADING-AGENT
```

### 2. Install Dependencies

#### Option A: Quick Installation
```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation (with TA-Lib)
```bash
# Install Python packages
pip install -r requirements.txt
pip install shap pyarrow fastparquet
```

#### Windows TA-Lib Installation
```bash
# For Windows users
pip install ta-lib
```

### 3. Prepare Your Data

Place your historical data files in the `/data` directory with the following format:

```csv
timestamp,open,high,low,close,volume
2016-05-05 00:00,1282.699,1282.890,1282.679,1282.679,73
2016-05-05 00:01,1282.681,1282.760,1282.679,1282.760,47
```

### 4. Configure Paths

Update the data file paths in `config/paths.py`:

```python
# Training data file
TRAIN_DATA_FILE = "data/XAUUSDM1-FULL.csv"

# Backtesting data file
BACKTEST_DATA_FILE = "data/XAUUSDM1-FULL-TEST.csv"
```

## 🔄 Two-Stage Workflow

### Stage 1: Data Preprocessing (One-time)

Process your raw CSV data into efficient Parquet format:

```bash
python scripts/preprocess_data.py
```

This stage:
- Loads massive CSV files
- Performs advanced feature engineering
- Saves optimized Parquet files
- Reduces memory footprint for training

### Stage 2: Training & Analysis

Execute the complete training pipeline:

```bash
python main.py
```

This pipeline includes:
1. **Walk-Forward Training**: Multiple training windows
2. **Model Fine-tuning**: Optimization on recent data
3. **Feature Analysis**: SHAP and permutation importance
4. **Backtesting**: Performance evaluation on unseen data

## 📁 Project Structure

```
DRL_Model_DeepSeek/
├── 📁 config/              # Configuration files
│   ├── config.yaml         # Main configuration
│   ├── paths.py           # Data paths
│   └── logging.yaml       # Logging configuration
├── 📁 data/               # Data files
│   ├── XAUUSDM1-FULL.csv # Training data
│   └── XAUUSDM1-TEST.csv # Testing data
├── 📁 envs/               # Trading environment
│   └── trading_env.py     # Gymnasium environment
├── 📁 models/             # Model architectures
│   └── custom_policy.py   # CNN-Transformer model
├── 📁 scripts/            # Utility scripts
│   ├── preprocess_data.py # Data preprocessing
│   ├── train_agent.py     # Training script
│   ├── backtest_agent.py  # Backtesting
│   └── analyze_features.py # Feature analysis
├── 📁 utils/              # Utility functions
│   ├── custom_indicators.py # Technical indicators
│   ├── data_transformation.py # Data processing
│   └── metrics.py         # Performance metrics
├── 📁 tests/              # Unit tests
├── main.py               # Main execution script
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## ⚙️ Configuration

### Model Parameters

Edit `config/config.yaml` to customize:

```yaml
model:
  architecture: "cnn_transformer"
  cnn_channels: [64, 128, 256]
  transformer_heads: 8
  transformer_layers: 6
  dropout: 0.1

training:
  total_timesteps: 1000000
  learning_rate: 0.0003
  batch_size: 64
  walk_forward_splits: 5

trading:
  initial_balance: 10000.0
  transaction_cost: 0.0001
  max_position_size: 1.0
```

### Environment Settings

```python
# Trading environment parameters
ENV_CONFIG = {
    'window_size': 60,          # Lookback period
    'initial_balance': 10000,   # Starting capital
    'transaction_cost': 0.0001, # Trading fees
    'max_drawdown': 0.2,       # Risk limit
}
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

### Financial Metrics
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss estimation
- **Calmar Ratio**: Return-to-drawdown ratio
- **Volatility**: Standard deviation of returns
- **Beta**: Market correlation coefficient

## 🔍 Model Analysis

### Feature Importance

The system provides detailed feature analysis:

```bash
python scripts/analyze_features.py
```

Output includes:
- SHAP feature importance plots
- Permutation importance rankings
- Feature correlation analysis
- Model decision explanations

### Backtesting

Comprehensive backtesting with:

```bash
python scripts/backtest_agent.py
```

Features:
- Out-of-sample testing
- Performance visualization
- Trade analysis
- Risk assessment

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Available tests:
- Environment functionality
- Indicator calculations
- Data preprocessing
- Model architecture

## 📈 Usage Examples

### Basic Training

```python
from scripts.train_agent import TrainingPipeline

# Initialize training pipeline
trainer = TrainingPipeline(config_path="config/config.yaml")

# Execute walk-forward training
trainer.run_walk_forward_training()

# Fine-tune on recent data
trainer.fine_tune_model()
```

### Custom Environment

```python
from envs.trading_env import TradingEnvironment

# Create custom environment
env = TradingEnvironment(
    data_path="data/custom_data.csv",
    window_size=30,
    initial_balance=5000
)

# Use with any RL algorithm
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Aliw02/DRL-TRADING-AGENT.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: This software is for educational and research purposes only. Trading financial instruments involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- **Stable Baselines3** team for the excellent RL framework
- **PyTorch** community for the deep learning foundation
- **TA-Lib** developers for technical analysis tools
- **OpenAI** for reinforcement learning research

## 📞 Support

For questions, issues, or suggestions:

- 📧 Email: [aliweyabood@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/Aliw02/DRL-TRADING-AGENT/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Aliw02/DRL-TRADING-AGENT/discussions)

## 🔄 Updates

### Version 1.0.0 (Latest)
- Initial release with CNN-Transformer architecture
- Walk-forward training implementation
- Comprehensive feature engineering
- SHAP model interpretability
- Professional backtesting suite

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

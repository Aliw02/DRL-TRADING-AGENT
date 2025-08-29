import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO):
    """Setup logging configuration"""
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        
    return logging.getLogger(__name__)
    
def get_logger(name):
    """Get a logger with the given name"""
    return logging.getLogger(name)

# Create logs directory if it doesn't exist
Path("results/logs").mkdir(parents=True, exist_ok=True)

# config/init.py (ROBUST AND CORRECTED VERSION)

import yaml
import os
from pathlib import Path

class Config:
    """
    Configuration management class that robustly finds its config file.
    """
    def __init__(self):
        # --- CRITICAL FIX: Build the absolute path to the config file ---
        # This finds the directory where this 'init.py' file lives,
        # then looks for 'config.yaml' inside that same directory.
        # This makes it work regardless of the script's working directory.
        self.config_path = Path(__file__).parent / "config.yaml"
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from YAML file using the absolute path."""
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found at the expected path: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return config_data

    def get(self, key, default=None):
        """Get configuration value by key (e.g., 'training.learning_rate')."""
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
    
# Global configuration instance, accessible throughout the project
config = Config()
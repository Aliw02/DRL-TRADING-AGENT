# config/init.py
# FINAL, FLEXIBLE, AND PRODUCTION-GRADE CONFIGURATION LOADER

import yaml
from pathlib import Path

class Config:
    """
    A centralized and flexible configuration management class.
    It is instantiated with a specific config file path, allowing different
    configurations (e.g., for SAC, PPO, testing) to be managed seamlessly.
    """

    def __init__(self, config_path: str = 'config/config_sac.yaml'):
        """
        The constructor now correctly accepts a 'config_path' argument,
        making the class versatile for the entire pipeline.

        Args:
            config_path (str): The path to the YAML configuration file to load.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Loads the configuration from the specified YAML file.
        """
        if not self.config_path.is_file():
            raise FileNotFoundError(f"CRITICAL: Config file not found at the specified path: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return config_data

    def get(self, key: str, default=None):
        """
        Retrieves a configuration value using a dot-separated key (e.g., 'training.learning_rate').
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str):
        """ Allows dictionary-style access to the configuration. """
        return self.get(key)


# A default global instance for modules that don't require a specific config path.
# This is used for simple scripts or when a default configuration is sufficient.
# The main pipeline will create its own instances with specific paths.
config = Config()
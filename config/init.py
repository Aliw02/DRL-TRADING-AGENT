
# config/init.py (CORRECTED AND MORE FLEXIBLE)
import yaml
from pathlib import Path

class Config:

    def __init__(self, config_path: str = 'config/config_sac.yaml'):
        
        # --- CRITICAL FIX: The constructor now accepts a config_path ---
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        if not self.config_path.is_file():
            # Also check for the default path relative to this file's location
            default_path = Path(__file__).parent / "config.yaml"
            if default_path.is_file():
                self.config_path = default_path
            else:
                raise FileNotFoundError(f"Config file not found at the specified path: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return config_data

    def get(self, key, default=None):
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


# A default global instance for modules that don't need a specific config
config = Config()

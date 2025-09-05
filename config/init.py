# config/init.py (DYNAMIC CONFIG LOADER)
import yaml
from pathlib import Path
import sys, os

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.accelerator import DEVICE # Import the device detector

class Config:
    def __init__(self, base_config_path: str = 'config/config.yaml'):
        self.base_config_path = Path(base_config_path)
        self.config = self._load_and_merge_configs()

    def _load_yaml(self, path):
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _deep_merge(self, base, new):
        """Recursively merges new dictionary into base."""
        for key, value in new.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _load_and_merge_configs(self):
        # 1. Load the base configuration
        base_config = self._load_yaml(self.base_config_path)
        print(f"Loaded base configuration from: {self.base_config_path}")

        # 2. Determine which device-specific config to load
        if DEVICE == 'cuda':
            device_config_path = self.base_config_path.parent / 'config_gpu.yaml'
            print(f"✅ GPU detected. Loading and merging GPU-specific settings...")
        else:
            device_config_path = self.base_config_path.parent / 'config_cpu.yaml'
            print(f"⚠️ CPU detected. Loading and merging CPU-specific settings...")

        # 3. Load and merge the device-specific configuration
        device_config = self._load_yaml(device_config_path)
        merged_config = self._deep_merge(base_config, device_config)
        
        print("Configuration loaded successfully.")
        return merged_config

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
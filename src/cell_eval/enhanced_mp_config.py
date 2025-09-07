"""
Enhanced Multiprocessing Configuration

Configuration management and presets for the enhanced multiprocessing system.

Author: Enhanced Multiprocessing Integration
Version: 1.0.0
"""

import os
import json
import warnings
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MPConfig:
    """Multiprocessing configuration"""
    process_count: int = 4
    chunk_size: int = 1
    timeout: float = 30.0
    enable_monitoring: bool = True
    auto_tune: bool = True
    error_threshold: float = 0.1
    restart_threshold: int = 5
    workload_type: str = 'general'
    max_memory_per_process: Optional[float] = None  # GB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MPConfig':
        """Create from dictionary"""
        return cls(**data)


class ConfigManager:
    """Configuration manager for enhanced multiprocessing"""
    
    DEFAULT_CONFIGS = {
        'general': MPConfig(
            process_count=4,
            chunk_size=10,
            timeout=30.0,
            workload_type='general'
        ),
        'cpu_intensive': MPConfig(
            process_count=8,
            chunk_size=1,
            timeout=60.0,
            workload_type='cpu_intensive',
            max_memory_per_process=2.0
        ),
        'memory_intensive': MPConfig(
            process_count=2,
            chunk_size=5,
            timeout=45.0,
            workload_type='memory_intensive',
            max_memory_per_process=4.0
        ),
        'io_intensive': MPConfig(
            process_count=16,
            chunk_size=20,
            timeout=90.0,
            workload_type='io_intensive'
        ),
        'mixed': MPConfig(
            process_count=6,
            chunk_size=3,
            timeout=50.0,
            workload_type='mixed',
            max_memory_per_process=3.0
        )
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.cell_eval'
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / 'mp_config.json'
        
        # Load or create config
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict[str, MPConfig]:
        """Load configurations from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    return {
                        name: MPConfig.from_dict(config_data)
                        for name, config_data in data.items()
                    }
            except Exception as e:
                warnings.warn(f"Failed to load config: {e}, using defaults")
        
        return self.DEFAULT_CONFIGS.copy()
    
    def save_configs(self):
        """Save configurations to file"""
        try:
            data = {
                name: config.to_dict()
                for name, config in self.configs.items()
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save config: {e}")
    
    def get_config(self, name: str) -> MPConfig:
        """Get configuration by name"""
        return self.configs.get(name, self.DEFAULT_CONFIGS['general'])
    
    def set_config(self, name: str, config: MPConfig):
        """Set configuration"""
        self.configs[name] = config
        self.save_configs()
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        return list(self.configs.keys())
    
    def reset_to_defaults(self):
        """Reset all configurations to defaults"""
        self.configs = self.DEFAULT_CONFIGS.copy()
        self.save_configs()


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_mp_config(name: str = 'general') -> MPConfig:
    """Get multiprocessing configuration"""
    return get_config_manager().get_config(name)

def set_mp_config(name: str, config: MPConfig):
    """Set multiprocessing configuration"""
    get_config_manager().set_config(name, config)

def list_mp_configs() -> List[str]:
    """List available configurations"""
    return get_config_manager().list_configs()

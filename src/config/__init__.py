"""
Configuration module for loading base and model configurations.

DEPRECATED: This module previously supported a two-file config system (base.yaml + models/*.yaml).
That system has been replaced with single-file configs (base_*.yaml).

All training scripts now use base_*.yaml files directly:
- base_distilbert.yaml
- base_indicbert.yaml
- base_indicbert_hpa.yaml
- base_xlmroberta.yaml
- base_mdeberta.yaml

Legacy configs have been moved to _legacy/ directory.
"""

import yaml
import os
import warnings
from pathlib import Path
from typing import Dict, Any

# Legacy base config path (deprecated)
_legacy_base_config_path = Path(__file__).parent / '_legacy' / 'base.yaml'
_base_config_path = Path(__file__).parent / 'base.yaml'  # Check root first for backward compat

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    DEPRECATED: Load model configuration by name from legacy models/ directory.
    
    This function is deprecated. Use base_*.yaml configs directly instead.
    
    Args:
        model_name: Model name (e.g., 'distilbert', 'indicbert_hpa')
    
    Returns:
        Model configuration dictionary
    
    Raises:
        FileNotFoundError: If legacy config not found
    """
    warnings.warn(
        "get_model_config() is deprecated. Use base_*.yaml configs directly. "
        "Legacy configs have been moved to _legacy/ directory.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Try legacy location first
    legacy_model_config_path = Path(__file__).parent / '_legacy' / 'models' / f'{model_name}.yaml'
    model_config_path = Path(__file__).parent / 'models' / f'{model_name}.yaml'
    
    if legacy_model_config_path.exists():
        with open(legacy_model_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(
            f"Legacy model config not found: {model_config_path} or {legacy_model_config_path}. "
            f"Please use base_*.yaml configs instead."
        )

# Legacy base_config (deprecated - only for backward compatibility)
_base_config = None
if _legacy_base_config_path.exists():
    with open(_legacy_base_config_path, 'r', encoding='utf-8') as f:
        _base_config = yaml.safe_load(f)
elif _base_config_path.exists():
    warnings.warn(
        "base.yaml in root config directory is deprecated. Use base_*.yaml configs instead.",
        DeprecationWarning,
        stacklevel=2
    )
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        _base_config = yaml.safe_load(f)
else:
    # Fallback default config (minimal)
    _base_config = {
        'paths': {
            'data_raw': 'data/raw',
            'data_processed': 'data/processed',
            'experiments': 'experiments'
        },
        'data': {
            'languages': ['en', 'hi', 'pa'],
            'max_seq_len': 256
        }
    }

base_config = _base_config  # For backward compatibility (deprecated)

__all__ = ['base_config', 'get_model_config']  # Deprecated exports


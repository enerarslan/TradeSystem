"""
Configuration module for AlphaTrade system.

This module provides centralized configuration management including:
- Global settings (settings.py)
- Trading configuration (trading_config.yaml)
- Logging configuration (logging_config.yaml)
"""

from pathlib import Path

import yaml

from config.settings import (
    PROJECT_ROOT,
    CONFIG_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    REPORTS_DIR,
    Settings,
    get_settings,
    reload_settings,
    settings,
)


def load_yaml_config(config_name: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_name: Name of the config file (with or without .yaml extension)

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    config_path = CONFIG_DIR / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_trading_config() -> dict:
    """Load the trading configuration."""
    return load_yaml_config("trading_config")


def load_logging_config() -> dict:
    """Load the logging configuration."""
    return load_yaml_config("logging_config")


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "CACHE_DIR",
    "REPORTS_DIR",
    "Settings",
    "get_settings",
    "reload_settings",
    "settings",
    "load_yaml_config",
    "load_trading_config",
    "load_logging_config",
]

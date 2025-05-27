"""Configuration loader utilities for YAML files."""

import os
import yaml
from typing import Dict, Any


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file and return its content as a dictionary.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Dictionary containing the YAML file content

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

        # Ensure we return a dictionary (yaml.safe_load can return None for empty files)
        if config_data is None:
            return {}

        return config_data

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error reading config file {config_path}: {e}")

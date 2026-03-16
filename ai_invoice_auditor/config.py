"""Configuration loader for rules.yaml."""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.yaml"

_config: dict | None = None


def get_config() -> dict:
    """Lazy-load and cache the rules.yaml configuration.

    Returns:
        Parsed YAML configuration as a dictionary.
    """
    global _config
    if _config is None:
        with open(_CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    return _config

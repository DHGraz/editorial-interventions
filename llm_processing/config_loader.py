"""
Configuration loader for LLM processing.

By default the pipeline uses config.py next to llm_processing.py. A project can
instead keep its own configuration file outside this folder and pass it with
--config (or the LLM_PROCESSING_CONFIG environment variable). The file is
registered as the module "config", so every "import config" in the processors
and utilities receives the project configuration.
"""

# Standard library imports
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

CONFIG_ENV_VAR = "LLM_PROCESSING_CONFIG"


def _config_path_from_argv() -> Optional[str]:
    """
    Read the --config option from the command line without consuming other options.

    Returns:
        The given path, or None if --config was not used.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", "-c")
    args, _ = parser.parse_known_args()
    return args.config


def load_config() -> ModuleType:
    """
    Load the configuration module and register it as "config".

    The path is taken from --config, then from the LLM_PROCESSING_CONFIG
    environment variable. Without either, the default config.py is imported.
    The directory of an external configuration file is appended to sys.path so
    that it can import shared settings (e.g. "from _common import *") without
    shadowing the modules of this package.

    Returns:
        The loaded configuration module.

    Raises:
        FileNotFoundError: If the given configuration file does not exist.
    """
    if "config" in sys.modules:
        return sys.modules["config"]

    config_path = _config_path_from_argv() or os.environ.get(CONFIG_ENV_VAR)
    if not config_path:
        import config
        return config

    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config_dir = str(path.parent)
    if config_dir not in sys.path:
        sys.path.append(config_dir)

    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = module
    spec.loader.exec_module(module)
    return module

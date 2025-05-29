import os
import sys
import yaml
import json
import joblib
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
from gaspriceforecast.utils.logger import get_logger

logger = get_logger(log_file="utils.log")


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads YAML file and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file

    Raises:
        ValueError: If YAML is empty or invalid
        Exception: For any unexpected issues

    Returns:
        ConfigBox: Dot-accessible config
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("YAML file is empty")
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is invalid or empty")
    except Exception as e:
        logger.error(f"Failed to load YAML file: {e}")
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Create a list of directories if they don't exist.

    Args:
        path_to_directories (list): List of paths (str or Path)
        verbose (bool): Log each created path if True
    """
    for path in path_to_directories:
        try:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"created directory at: {path}")
        except Exception as e:
            logger.error(f"failed to create directory {path}: {e}")
            raise e


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save a dictionary to a JSON file.

    Args:
        path (Path): Path where to save
        data (dict): Data to save
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"json file saved at: {path}")
    except Exception as e:
        logger.error(f"failed to save JSON at {path}: {e}")
        raise e


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a JSON file and return as ConfigBox.

    Args:
        path (Path): Path to JSON file

    Returns:
        ConfigBox: Dot-access config object
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"failed to load JSON from {path}: {e}")
        raise e


@ensure_annotations
def save_model(path: Path, model: Any):
    """
    Save a model using joblib.

    Args:
        path (Path): Path to save the model
        model (Any): Model object
    """
    try:
        joblib.dump(value=model, filename=path)
        logger.info(f"model saved at: {path}")
    except Exception as e:
        logger.error(f"failed to save model at {path}: {e}")
        raise e


@ensure_annotations
def load_model(path: Path) -> Any:
    """
    Load a joblib model from path.

    Args:
        path (Path): Path to model

    Returns:
        Any: Loaded model object
    """
    try:
        model = joblib.load(path)
        logger.info(f"model loaded from: {path}")
        return model
    except Exception as e:
        logger.error(f"failed to load model from {path}: {e}")
        raise e

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    If the configuration file has an 'inherit' key, it will recursively load the parent configuration
    and merge the child configuration on top of it.
    """
    path_obj = Path(config_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path_obj}")

    config = load_yaml(str(path_obj))

    if 'inherit' in config:
        parent_path = path_obj.parent / config['inherit']
        parent_config = load_config(str(parent_path.resolve()))
        config = deep_update(parent_config, config)
        del config['inherit']

    return config

def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively updates a nested dictionary.
    """
    import copy
    result = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result

if __name__ == "__main__":
    # Test logic
    base_cfg = load_config("configs/base_config.yaml")
    print("Base config loaded successfully.")

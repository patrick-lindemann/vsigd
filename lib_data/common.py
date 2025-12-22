from pathlib import Path
from typing import Dict

import yaml


def parse_yaml_file(filepath: Path) -> Dict:
    assert filepath.exists(), f"Non-existent yaml file: {filepath}"
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data


def write_yaml_file(data: Dict, filepath: Path):
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

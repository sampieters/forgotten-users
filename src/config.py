"""
config.py – Global configuration, constants, and paths
=======================================================
All other modules import their constants from here.  Nothing else in the
project should read the config file directly.
"""

from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH = Path("data/raw")
CHECKPOINT_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
CONFIG_PATH = Path("config/config.yaml")

# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_datasets_config(config: dict) -> tuple[list[str], dict]:
    """Parse the 'datasets' section of config.yaml.

    Supports dictionary mappings, lists of dicts, or simple lists of dataset
    names.

    Returns
    -------
    dataset_names : list[str]
        Active dataset name strings.
    dataset_configs : dict
        Mapping of ``dataset_name -> {"cold_start": int, "preprocessing": list}``.
    """
    raw_datasets = config.get("datasets", ["MovieLens100K"])
    dataset_names: list[str] = []
    dataset_configs: dict = {}

    if isinstance(raw_datasets, dict):
        for ds_name, ds_info in raw_datasets.items():
            dataset_names.append(ds_name)
            if isinstance(ds_info, dict):
                dataset_configs[ds_name] = {
                    "cold_start": ds_info.get("cold_start", 20),
                    "preprocessing": ds_info.get("preprocessing", []),
                }
            else:
                dataset_configs[ds_name] = {"cold_start": 20, "preprocessing": []}

    elif isinstance(raw_datasets, list):
        for item in raw_datasets:
            if isinstance(item, str):
                dataset_names.append(item)
                dataset_configs[item] = {"cold_start": 20, "preprocessing": []}
            elif isinstance(item, dict):
                ds_name = item.get("name")
                if ds_name:
                    dataset_names.append(ds_name)
                    dataset_configs[ds_name] = {
                        "cold_start": item.get("cold_start", 20),
                        "preprocessing": item.get("preprocessing", []),
                    }

    return dataset_names, dataset_configs


# ---------------------------------------------------------------------------
# Load config & expose module-level constants
# ---------------------------------------------------------------------------

CONFIG = load_config(CONFIG_PATH)

DATASETS, DATASET_CONFIGS = parse_datasets_config(CONFIG)
K: int = CONFIG.get("evaluation", {}).get("k", 10)
MODEL_SEEDS: list[int] = CONFIG.get("evaluation", {}).get("model_seeds", [42])
SPLIT_SEEDS: list[int] = CONFIG.get("evaluation", {}).get("split_seeds", [42])
ALGORITHM_CONFIGS: dict = CONFIG.get("algorithms", {})

# Splitting ratios & caps
TRAIN_FRAC: float = 0.8
VAL_FRAC: float = 0.8
COREPRUNING: int = 5
HOLDOUT: int = 5


def get_cold_start(ds_name: str) -> int:
    """Return the cold-start history-length threshold for the given dataset."""
    return DATASET_CONFIGS.get(ds_name, {}).get("cold_start", 20)

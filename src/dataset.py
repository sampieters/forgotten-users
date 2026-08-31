"""
dataset.py – Dataset registry and loader
=========================================
Maps dataset names to their RecPack classes and applies preprocessing
filters as specified in the configuration.
"""

from recpack.datasets.base import Dataset
from recpack.datasets import (
    MovieLens100K,
    MovieLens1M,
    CiteULike,
    Netflix,
)
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.filters import (
    Deduplicate,
    MinItemsPerUser,
    MinRating,
    MinUsersPerItem,
)
from config import COREPRUNING, DATASET_CONFIGS, DATASET_PATH
import pandas as pd
import json

# ---------------------------------------------------------------------------
# Amazon Prime Pantry dataset implementation
# ---------------------------------------------------------------------------

class Amazon2018(Dataset):
    USER_IX = "reviewerID"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "asin"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "unixReviewTime"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    RATING_IX = "overall"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""

    @property
    def DEFAULT_FILENAME(self) -> str:
        """Default filename that will be used if it is not specified by the user."""
        return f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}"

class PrimePantry(Amazon2018):
    REMOTE_FILENAME = "Prime_Pantry.json"
    REMOTE_ZIPNAME = "primepantry"

    USER_IX = "reviewerID"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "asin"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "unixReviewTime"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    RATING_IX = "overall"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""

    @property
    def DEFAULT_FILENAME(self) -> str:
        """Default filename that will be used if it is not specified by the user."""
        return f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}"

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from a JSON file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        # Load JSON objects line by line
        with open(self.file_path, 'r') as f:
            data = [json.loads(line) for line in f]

        df = pd.DataFrame(data)

        # Only keep the relevant columns
        df = df[[self.USER_IX, self.ITEM_IX, self.TIMESTAMP_IX, self.RATING_IX]]

        self.df = df
        return df

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict = {
    "MovieLens100K": MovieLens100K,
    "MovieLens1M": MovieLens1M,
    "PrimePantry": PrimePantry,
    "CiteULike": CiteULike,
    "Netflix": Netflix,
}

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_dataset(dataset_name: str) -> InteractionMatrix:
    """Load and preprocess a dataset according to the config.

    Falls back to core-pruning + deduplication when no preprocessing steps
    are specified in the config.
    """
    dataset_cls = DATASET_REGISTRY[dataset_name]
    loader = dataset_cls(path=DATASET_PATH, use_default_filters=False)

    preprocessing_steps = DATASET_CONFIGS.get(dataset_name, {}).get("preprocessing", [])

    if preprocessing_steps:
        for step in preprocessing_steps:
            if isinstance(step, str):
                filter_type, params = step, {}
            elif isinstance(step, dict):
                filter_type = step.get("type")
                params = step.get("params", {})
            else:
                continue

            if filter_type == "MinRating":
                min_rating = params.get("min_rating", params.get("threshold", 3))
                loader.add_filter(MinRating(min_rating, loader.RATING_IX))

            elif filter_type == "MinItemsPerUser":
                min_items = params.get(
                    "min_items_per_user",
                    params.get("min_items", params.get("core", COREPRUNING)),
                )
                loader.add_filter(MinItemsPerUser(min_items, loader.ITEM_IX, loader.USER_IX))

            elif filter_type == "MinUsersPerItem":
                min_users = params.get(
                    "min_users_per_item",
                    params.get("min_users", params.get("core", COREPRUNING)),
                )
                loader.add_filter(MinUsersPerItem(min_users, loader.ITEM_IX, loader.USER_IX))

            elif filter_type == "Deduplicate":
                loader.add_filter(Deduplicate(loader.ITEM_IX, loader.USER_IX))
    else:
        # Default preprocessing when none is specified in config.
        loader.add_filter(MinItemsPerUser(COREPRUNING, loader.ITEM_IX, loader.USER_IX))
        loader.add_filter(MinUsersPerItem(COREPRUNING, loader.ITEM_IX, loader.USER_IX))
        loader.add_filter(Deduplicate(loader.ITEM_IX, loader.USER_IX))

    return loader.load()

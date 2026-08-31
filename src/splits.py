"""
splits.py – Train/validation/test split construction and prediction utilities
=============================================================================
"""

import pickle
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from scipy.sparse import csr_matrix

from recpack.matrix import InteractionMatrix
from recpack.scenarios.splitters import (
    FractionInteractionSplitter,
    StrongGeneralizationSplitter,
    Splitter,
)

from config import (
    CHECKPOINT_DIR,
    HOLDOUT,
    MODEL_SEEDS,
    SPLIT_SEEDS,
    TRAIN_FRAC,
    VAL_FRAC,
)

logger = logging.getLogger("recpack")

# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

class CountInteractionSplitter(Splitter):
    """Split data randomly, such that ``n_interactions``
    of interactions are assigned to the first return value and the remainder to the second.

    :param n_interactions: Number of events to end up in the first return value.
    :type n_interactions: int
    :param seed: Seed the random generator. Set this value
        if you require reproducible results.
        Defaults to None, which results in a random seed.
    :type seed: int, optional
    """

    def __init__(self, n_interactions, seed: int = None):
        super().__init__()
        self.n_interactions = n_interactions

        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]

        self.seed = seed

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        in_interactions = []
        out_interactions = []

        for u, interaction_history in tqdm(data.interaction_history):

            rstate = np.random.RandomState(self.seed + u)

            interaction_history = interaction_history.copy()
            rstate.shuffle(interaction_history)

            hist_len = len(interaction_history)

            # Number of interactions to place in the test set
            n_out = min(self.n_interactions, hist_len)

            cut = hist_len - n_out

            in_interactions.extend(interaction_history[:cut])
            out_interactions.extend(interaction_history[cut:])

        data_in = data.interactions_in(in_interactions)
        data_out = data.interactions_in(out_interactions)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out


def build_splits(ds_name: str, interaction_matrix: InteractionMatrix) -> None:
    """Create and persist train/validation/test splits for all seeds.

    The function is idempotent: existing split files are skipped.
    """
    # --- Training & validation splits (one per model seed) ---
    for model_seed in MODEL_SEEDS:
        train_dir = Path(CHECKPOINT_DIR) / ds_name / f"model_{model_seed}" / "data"
        train_dir.mkdir(parents=True, exist_ok=True)
        train_path = train_dir / "train.pkl"

        if train_path.exists():
            print(f"{train_path} already exists, skipping.")
            continue

        strong_gen = StrongGeneralizationSplitter(TRAIN_FRAC, seed=model_seed)
        full_train_X, _ = strong_gen.split(interaction_matrix)

        validation_splitter = StrongGeneralizationSplitter(VAL_FRAC, seed=model_seed)
        val_train_X, validation_data = validation_splitter.split(full_train_X)

        interaction_split = FractionInteractionSplitter(VAL_FRAC, seed=model_seed)
        val_data_in, val_data_out = interaction_split.split(validation_data)

        with open(train_path, "wb") as f:
            pickle.dump(full_train_X, f)
            pickle.dump(val_train_X, f)
            pickle.dump(val_data_in, f)
            pickle.dump(val_data_out, f)

    # --- Test splits (one per model seed x split seed) ---
    for model_seed in MODEL_SEEDS:
        strong_gen = StrongGeneralizationSplitter(TRAIN_FRAC, seed=model_seed)
        full_train_X, test_data = strong_gen.split(interaction_matrix)

        for split_seed in SPLIT_SEEDS:
            test_path = (
                Path(CHECKPOINT_DIR)
                / ds_name
                / f"model_{model_seed}"
                / "data"
                / f"split_{split_seed}"
            )
            if test_path.exists():
                print(f"{test_path} already exists, skipping.")
                continue

            interaction_split = CountInteractionSplitter(HOLDOUT, seed=split_seed)
            test_data_in, test_data_out = interaction_split.split(test_data)

            holdout_counts = np.diff(test_data_out.values.indptr)
            assert np.all(holdout_counts <= HOLDOUT), (
                f"Found users with more than {HOLDOUT} held-out items. "
                f"Maximum: {holdout_counts.max()}"
            )

            test_path.mkdir(parents=True, exist_ok=True)
            with open(test_path / "test_data.pkl", "wb") as f:
                pickle.dump(test_data_in, f)
                pickle.dump(test_data_out, f)


# ---------------------------------------------------------------------------
# Prediction utilities
# ---------------------------------------------------------------------------


def truncate_topk(X: csr_matrix, k: int) -> csr_matrix:
    """Return a copy of *X* keeping only the top-*k* scores per row."""
    data, indices, indptr = [], [], [0]

    for u in range(X.shape[0]):
        start, end = X.indptr[u], X.indptr[u + 1]
        row_data = X.data[start:end]
        row_indices = X.indices[start:end]

        if len(row_data) > k:
            top = np.argpartition(-row_data, k - 1)[:k]
            top = top[np.argsort(-row_data[top])]
        else:
            top = np.argsort(-row_data)

        data.extend(row_data[top])
        indices.extend(row_indices[top])
        indptr.append(len(data))

    return csr_matrix(
        (np.array(data), np.array(indices), np.array(indptr)),
        shape=X.shape,
    )

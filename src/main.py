"""
main.py – Forgotten Users Experiment Pipeline
=================================================
Phases
------
1. Build datasets, train/validation/test splits, and model predictions.
2. Select hyperparameters and evaluate models.
3. Generate diagnostic plots (hit-rate, NDCG, UMAP) for forgotten users.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from scipy.sparse import csr_matrix
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle

from config import (
    DATASETS, K, MODEL_SEEDS, SPLIT_SEEDS, ALGORITHM_CONFIGS, 
    TRAIN_FRAC, VAL_FRAC, COREPRUNING, HOLDOUT, DATASET_CONFIGS,
    CHECKPOINT_DIR, load_config, parse_datasets_config, get_cold_start
)

from recpack.pipelines.registries import ALGORITHM_REGISTRY
from recpack.algorithms.base import TorchMLAlgorithm
from recpack.pipelines import PipelineBuilder
from recpack.matrix import InteractionMatrix
from recpack.metrics import HitK, NDCGK

from plots import (
    plot_forgotten_users, 
    plot_umap_forgotten_users,
    write_forgotten_users_statistics, 
    plot_candidate_pooling_forgotten_users
)

from dataset import DATASET_REGISTRY, build_dataset
from splits import build_splits, truncate_topk
from StrongGenSVD import StrongGenSVD
# Add implemented SVD algortihm for Strong Generalization
ALGORITHM_REGISTRY.register(StrongGenSVD.__name__, StrongGenSVD)

# ---------------------------------------------------------------------------
# Main Experiment Pipeline
# ---------------------------------------------------------------------------

def run_experiment() -> None:
    """Execute the full experiment: data preparation, HP selection, and prediction."""
    print("\n" + "=" * 70)
    print("PHASE 1: Building models and splits")
    print("=" * 70)

    for ds_name in DATASETS:
        print(f"Building dataset for: {ds_name}")
        Path(CHECKPOINT_DIR / ds_name).mkdir(parents=True, exist_ok=True)
        interaction_matrix = build_dataset(ds_name)

        print(f"Building splits for: {ds_name}")
        build_splits(ds_name, interaction_matrix)

    print("Building models and splits done.")

    print("\n" + "=" * 70)
    print("PHASE 2: Hyperparameter selection & evaluation")
    print("=" * 70)

    builder = PipelineBuilder()
    for model_name, model_config in ALGORITHM_CONFIGS.items():
        builder.add_algorithm(model_name, grid=model_config)

    builder.set_optimisation_metric("NDCGK", K=K)
    builder.add_metric("NDCGK", K=K)

    for ds_name in DATASETS:
        for model_seed in MODEL_SEEDS:
            train_path = (
                Path(CHECKPOINT_DIR) / ds_name / f"model_{model_seed}" / "data" / "train.pkl"
            )
            with open(train_path, "rb") as f:
                full_train_X = pickle.load(f)
                val_train_X = pickle.load(f)
                val_data_in = pickle.load(f)
                val_data_out = pickle.load(f)

            builder.set_full_training_data(full_train_X)
            builder.set_validation_training_data(val_train_X)
            builder.set_validation_data((val_data_in, val_data_out))

            mock_test = csr_matrix(full_train_X.shape)
            builder.set_test_data((mock_test, mock_test))
            pipeline = builder.build()

            for algorithm_entry in tqdm(pipeline.algorithm_entries):
                if algorithm_entry.optimise:
                    print(f"Hyperparameter selection for: {ds_name}, {algorithm_entry.name}")
                    params = pipeline._optimise_hyperparameters(algorithm_entry)
                else:
                    params = algorithm_entry.params

                print(
                    f"Hyperparameters found for: {ds_name}, "
                    f"{algorithm_entry.name}: {params}"
                )
                algorithm = ALGORITHM_REGISTRY.get(algorithm_entry.name)(**params)

                training_data = (
                    pipeline.validation_training_data
                    if isinstance(algorithm, TorchMLAlgorithm)
                    else pipeline.full_training_data
                )
                pipeline._train(algorithm, training_data)

                hyper_path = (
                    Path(CHECKPOINT_DIR)
                    / ds_name
                    / f"model_{model_seed}"
                    / "algorithms"
                    / algorithm_entry.name
                )
                hyper_path.mkdir(parents=True, exist_ok=True)
                with open(hyper_path / "hyperparameters.pkl", "wb") as f:
                    pickle.dump(params, f)

                print(
                    f"Running splits for best hyperparameters: "
                    f"{ds_name}, {algorithm_entry.name}"
                )

                for split_id in SPLIT_SEEDS:
                    pred_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "algorithms"
                        / algorithm_entry.name
                        / f"split_{split_id}"
                        / "predictions.pkl"
                    )
                    if pred_path.exists():
                        print(f"Predictions for split {split_id} already exist, skipping.")
                        continue

                    test_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "data"
                        / f"split_{split_id}"
                        / "test_data.pkl"
                    )
                    with open(test_path, "rb") as f:
                        test_data_in = pickle.load(f)
                        test_data_out = pickle.load(f)

                    pipeline.test_data_in = test_data_in
                    pipeline.test_data_out = test_data_out

                    X_pred = pipeline._predict_and_postprocess(algorithm, pipeline.test_data_in)
                    X_pred = truncate_topk(X_pred, k=100)

                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(pred_path, "wb") as f:
                        pickle.dump(X_pred, f)

if __name__ == "__main__":
    #run_experiment()
    plot_forgotten_users()
    plot_umap_forgotten_users()
    write_forgotten_users_statistics()
    plot_candidate_pooling_forgotten_users()


from recpack.metrics import HitK, NDCGK
import matplotlib.pyplot as plt
import umap.umap_ as umap
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

from config import DATASETS, K, MODEL_SEEDS, SPLIT_SEEDS, CONFIG, ALGORITHM_CONFIGS, CHECKPOINT_DIR, RESULTS_DIR
from config import get_cold_start

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _result_dir(ds_name: str, model_seed: int, model_name: str) -> Path:
    """Return (and create) the output directory for a given experiment run."""
    out = Path(RESULTS_DIR) / ds_name / f"model_{model_seed}" / model_name
    out.mkdir(parents=True, exist_ok=True)
    return out

# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def plot_forgotten_users() -> None:
    """Scatter plot of hit-count difference between a model and popularity."""
    for ds_name in DATASETS:
        for model_seed in MODEL_SEEDS:
            for model_name in ALGORITHM_CONFIGS:
                if model_name.lower() == "popularity":
                    continue

                model_hit_counts = None
                pop_hit_counts = None
                user_history_lengths = None
                user_ids = None
                avg_ndcg = 0.0
                avg_pop_ndcg = 0.0

                for split_id in SPLIT_SEEDS:
                    pred_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "algorithms"
                        / model_name
                        / f"split_{split_id}"
                        / "predictions.pkl"
                    )
                    with open(pred_path, "rb") as f:
                        model_pred = pickle.load(f)

                    pop_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "algorithms"
                        / "Popularity"
                        / f"split_{split_id}"
                        / "predictions.pkl"
                    )
                    with open(pop_path, "rb") as f:
                        pop_pred = pickle.load(f)

                    test_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "data"
                        / f"split_{split_id}"
                        / "test_data.pkl"
                    )
                    with open(test_path, "rb") as f:
                        test_in = pickle.load(f)
                        test_out = pickle.load(f)

                    hit_metric = HitK(K=K)

                    hit_metric.calculate(test_out.binary_values, model_pred)
                    model_hits = (
                        hit_metric.results.groupby("user_id")["score"]
                        .any()
                        .astype(int)
                        .reset_index()
                    )

                    hit_metric.calculate(test_out.binary_values, pop_pred)
                    pop_hits = (
                        hit_metric.results.groupby("user_id")["score"]
                        .any()
                        .astype(int)
                        .reset_index()
                    )

                    ndcg = NDCGK(K=K)
                    ndcg.calculate(test_out.binary_values, model_pred)
                    avg_ndcg += ndcg.value
                    ndcg.calculate(test_out.binary_values, pop_pred)
                    avg_pop_ndcg += ndcg.value

                    if user_ids is None:
                        user_ids = model_hits["user_id"].values
                        model_hit_counts = np.zeros(len(user_ids), dtype=int)
                        pop_hit_counts = np.zeros(len(user_ids), dtype=int)

                        active_users = sorted(test_in.active_users)
                        hist = test_in.values[active_users].getnnz(axis=1)
                        user_history_lengths = pd.Series(hist, index=active_users)

                    model_hit_counts += model_hits["score"].values
                    pop_hit_counts += pop_hits["score"].values

                avg_ndcg /= len(SPLIT_SEEDS)
                avg_pop_ndcg /= len(SPLIT_SEEDS)

                df = pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "history_length": user_history_lengths.reindex(user_ids).values,
                        "model_hits": model_hit_counts,
                        "pop_hits": pop_hit_counts,
                    }
                )
                df["hit_difference"] = df["model_hits"] - df["pop_hits"]

                cold_start_threshold = get_cold_start(ds_name)
                cold_start_mask = df["history_length"] <= cold_start_threshold
                worse_mask = df["hit_difference"] < 0
                cold_normal_mask = cold_start_mask & ~worse_mask
                forgotten_mask = (~cold_start_mask) & worse_mask
                normal_mask = (~cold_start_mask) & ~worse_mask

                fig, ax = plt.subplots(figsize=(8, 5))

                if cold_start_mask.any():
                    ax.scatter(
                        df.loc[cold_start_mask, "history_length"],
                        df.loc[cold_start_mask, "hit_difference"],
                        color="grey", marker="s", edgecolors="none", alpha=0.7, s=12,
                        label=f"Cold-start users (n={cold_start_mask.sum()})",
                    )

                if normal_mask.any():
                    ax.scatter(
                        df.loc[normal_mask, "history_length"],
                        df.loc[normal_mask, "hit_difference"],
                        color="grey", marker="o", edgecolors="none", alpha=0.7, s=12,
                        label=f"Regular users (n={normal_mask.sum()})",
                    )

                if forgotten_mask.any():
                    ax.scatter(
                        df.loc[forgotten_mask, "history_length"],
                        df.loc[forgotten_mask, "hit_difference"],
                        color="orange", marker="o", edgecolors="none", alpha=0.7, s=12,
                        label=f"Forgotten users (n={forgotten_mask.sum()})",
                    )

                ax.axhline(
                    0, color="black", linestyle="--", linewidth=1,
                )
                ax.axvline(get_cold_start(ds_name), color="gray", linestyle=":", linewidth=1)

                # Fixed y-axis range
                ax.set_ylim(-20.5, 20.5)

                ax.set_xlabel("History Length")
                ax.set_ylabel("Hit Difference")
                ax.legend(loc="upper right")

                plt.savefig(
                    _result_dir(ds_name, model_seed, model_name) / "forgotten_users.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)


def plot_umap_forgotten_users() -> None:
    """UMAP embedding coloured by user group using the popularity baseline."""

    for ds_name in DATASETS:
        for model_seed in MODEL_SEEDS:
            train_path = (
                Path(CHECKPOINT_DIR)
                / ds_name
                / f"model_{model_seed}"
                / "data"
                / "train.pkl"
            )

            with open(train_path, "rb") as f:
                full_train_X = pickle.load(f)

            for model_name in ALGORITHM_CONFIGS:
                if model_name.lower() == "popularity":
                    continue

                model_hit_counts = None
                pop_hit_counts = None
                user_history_lengths = None
                user_ids = None

                for split_id in SPLIT_SEEDS:
                    # Model predictions
                    model_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "algorithms"
                        / model_name
                        / f"split_{split_id}"
                        / "predictions.pkl"
                    )

                    with open(model_path, "rb") as f:
                        model_pred = pickle.load(f)

                    # Popularity predictions
                    pop_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "algorithms"
                        / "Popularity"
                        / f"split_{split_id}"
                        / "predictions.pkl"
                    )

                    with open(pop_path, "rb") as f:
                        pop_pred = pickle.load(f)

                    # Test data
                    test_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "data"
                        / f"split_{split_id}"
                        / "test_data.pkl"
                    )

                    with open(test_path, "rb") as f:
                        test_in = pickle.load(f)
                        test_out = pickle.load(f)

                    hit_metric = HitK(K=K)

                    # Model hits
                    hit_metric.calculate(test_out.binary_values, model_pred)
                    model_hits = (
                        hit_metric.results.groupby("user_id")["score"]
                        .any()
                        .astype(int)
                        .reset_index()
                    )

                    # Popularity hits
                    hit_metric.calculate(test_out.binary_values, pop_pred)
                    pop_hits = (
                        hit_metric.results.groupby("user_id")["score"]
                        .any()
                        .astype(int)
                        .reset_index()
                    )

                    if user_ids is None:
                        user_ids = model_hits["user_id"].values

                        model_hit_counts = np.zeros(len(user_ids), dtype=int)
                        pop_hit_counts = np.zeros(len(user_ids), dtype=int)

                        active_users = sorted(test_in.active_users)
                        hist = test_in.values[active_users].getnnz(axis=1)
                        user_history_lengths = pd.Series(hist, index=active_users)

                    model_hit_counts += model_hits["score"].values
                    pop_hit_counts += pop_hits["score"].values

                df = pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "history_length": user_history_lengths.reindex(user_ids).values,
                        "model_hits": model_hit_counts,
                        "pop_hits": pop_hit_counts,
                    }
                )

                cold_threshold = get_cold_start(ds_name)

                cold_mask = df["history_length"].values <= cold_threshold
                non_cold_mask = ~cold_mask

                pop_forgotten = (
                    df["model_hits"].values < df["pop_hits"].values
                )

                # Final groups
                forgotten_mask = pop_forgotten & non_cold_mask
                normal_mask = ~pop_forgotten

                user_matrix = full_train_X.values[user_ids]

                # Use n_neighbors=750 for Netflix dataset
                reducer = umap.UMAP(
                    n_neighbors=750,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )

                embedding = reducer.fit_transform(user_matrix)

                fig, ax = plt.subplots(figsize=(5.2, 4.2))

                if normal_mask.any():
                    ax.scatter(
                        embedding[normal_mask, 0],
                        embedding[normal_mask, 1],
                        color="grey",
                        s=12,
                        marker="o",
                        alpha=0.7,
                        linewidths=0,
                        label="Regular users",
                    )

                if forgotten_mask.any():
                    ax.scatter(
                        embedding[forgotten_mask, 0],
                        embedding[forgotten_mask, 1],
                        color="orange",
                        s=20,
                        marker="o",
                        alpha=0.7,
                        linewidths=0,
                        label="Forgotten users",
                    )

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")

                for spine in ax.spines.values():
                    spine.set_visible(False)

                ax.legend(loc="upper right", fontsize=8, frameon=True)
                plt.tight_layout()

                plt.savefig(
                    _result_dir(ds_name, model_seed, model_name)
                    / "forgotten_users_umap.png",
                    dpi=300,
                    bbox_inches="tight",
                )

                plt.close(fig)


def write_forgotten_users_statistics() -> None:
    """Write forgotten user statistics relative to popularity baseline."""

    for ds_name in DATASETS:
        for model_seed in MODEL_SEEDS:
            for model_name in ALGORITHM_CONFIGS:
                if model_name.lower() == "popularity":
                    continue

                model_hit_counts = None
                pop_hit_counts = None
                user_history_lengths = None
                user_ids = None

                for split_id in SPLIT_SEEDS:
                    # Load model predictions
                    model_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "algorithms"
                        / model_name
                        / f"split_{split_id}"
                        / "predictions.pkl"
                    )

                    with open(model_path, "rb") as f:
                        model_pred = pickle.load(f)

                    # Load popularity predictions
                    pop_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "algorithms"
                        / "Popularity"
                        / f"split_{split_id}"
                        / "predictions.pkl"
                    )

                    with open(pop_path, "rb") as f:
                        pop_pred = pickle.load(f)

                    # Load test data
                    test_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "data"
                        / f"split_{split_id}"
                        / "test_data.pkl"
                    )

                    with open(test_path, "rb") as f:
                        test_in = pickle.load(f)
                        test_out = pickle.load(f)

                    hit_metric = HitK(K=K)

                    # Model hits
                    hit_metric.calculate(test_out.binary_values, model_pred)
                    model_hits = (
                        hit_metric.results.groupby("user_id")["score"]
                        .any()
                        .astype(int)
                        .reset_index()
                    )

                    # Popularity hits
                    hit_metric.calculate(test_out.binary_values, pop_pred)
                    pop_hits = (
                        hit_metric.results.groupby("user_id")["score"]
                        .any()
                        .astype(int)
                        .reset_index()
                    )

                    if user_ids is None:
                        user_ids = model_hits["user_id"].values

                        model_hit_counts = np.zeros(
                            len(user_ids), dtype=int
                        )
                        pop_hit_counts = np.zeros(
                            len(user_ids), dtype=int
                        )

                        active_users = sorted(test_in.active_users)
                        hist = test_in.values[active_users].getnnz(axis=1)

                        user_history_lengths = pd.Series(
                            hist,
                            index=active_users,
                        )

                    model_hit_counts += model_hits["score"].values
                    pop_hit_counts += pop_hits["score"].values

                # Create dataframe
                df = pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "history_length": (
                            user_history_lengths
                            .reindex(user_ids)
                            .values
                        ),
                        "model_hits": model_hit_counts,
                        "pop_hits": pop_hit_counts,
                    }
                )

                # Compare against popularity baseline
                pop_forgotten_mask = (
                    df["model_hits"].values
                    < df["pop_hits"].values
                )

                # Split cold-start / non-cold-start
                cold_mask = (
                    df["history_length"].values
                    <= get_cold_start(ds_name)
                )

                non_cold_mask = ~cold_mask

                cold_forgotten = (
                    pop_forgotten_mask
                    & cold_mask
                )

                non_cold_forgotten = (
                    pop_forgotten_mask
                    & non_cold_mask
                )

                # \UserDefinition{} excludes cold-start users
                forgotten_mask = non_cold_forgotten

                # Counts
                n_users = len(df)

                n_forgotten = forgotten_mask.sum()

                n_cold = cold_mask.sum()
                n_non_cold = non_cold_mask.sum()

                n_cold_forgotten = cold_forgotten.sum()
                n_non_cold_forgotten = non_cold_forgotten.sum()

                # Percentages
                pct_forgotten = (
                    100 * n_forgotten / n_users
                    if n_users > 0
                    else 0.0
                )

                pct_cold_users = (
                    100 * n_cold / n_users
                    if n_users > 0
                    else 0.0
                )

                pct_cold_forgotten = (
                    100 * n_cold_forgotten / n_cold
                    if n_cold > 0
                    else 0.0
                )

                pct_non_cold_users = (
                    100 * n_non_cold / n_users
                    if n_users > 0
                    else 0.0
                )

                pct_non_cold_forgotten = (
                    100 * n_non_cold_forgotten / n_non_cold
                    if n_non_cold > 0
                    else 0.0
                )

                # Users where both the model and popularity fail completely
                zero_both_mask = (
                    (df["model_hits"].values == 0)
                    & (df["pop_hits"].values == 0)
                )

                zero_both_cold = zero_both_mask & cold_mask
                zero_both_non_cold = zero_both_mask & non_cold_mask

                n_zero_both = zero_both_mask.sum()
                n_zero_both_cold = zero_both_cold.sum()
                n_zero_both_non_cold = zero_both_non_cold.sum()

                pct_zero_both = (
                    100 * n_zero_both / n_users
                    if n_users > 0
                    else 0.0
                )

                pct_zero_both_cold = (
                    100 * n_zero_both_cold / n_cold
                    if n_cold > 0
                    else 0.0
                )

                pct_zero_both_non_cold = (
                    100 * n_zero_both_non_cold / n_non_cold
                    if n_non_cold > 0
                    else 0.0
                )

                # Write results
                out_file = (
                    _result_dir(ds_name, model_seed, model_name)
                    / "forgotten_user_statistics.txt"
                )

                with open(out_file, "w") as f:
                    f.write(f"Dataset: {ds_name}\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Seed: {model_seed}\n\n")

                    f.write(
                        f"Forgotten users (non-cold-start): "
                        f"{n_forgotten}/{n_users} "
                        f"({pct_forgotten:.2f}% of all users)\n\n"
                    )

                    f.write(
                        f"Cold-start users: "
                        f"{n_cold}/{n_users} "
                        f"({pct_cold_users:.2f}% of all users)\n"
                    )

                    f.write(
                        f"Cold-start forgotten: "
                        f"{n_cold_forgotten}/{n_cold} "
                        f"({pct_cold_forgotten:.2f}% of cold-start users)\n\n"
                    )

                    f.write(
                        f"Non-cold-start users: "
                        f"{n_non_cold}/{n_users} "
                        f"({pct_non_cold_users:.2f}% of all users)\n"
                    )

                    f.write(
                        f"Non-cold-start forgotten: "
                        f"{n_non_cold_forgotten}/{n_non_cold} "
                        f"({pct_non_cold_forgotten:.2f}% of non-cold-start users)\n"
                    )

                    f.write("\n")

                    f.write(
                        f"Users with zero hits for both model and popularity: "
                        f"{n_zero_both}/{n_users} "
                        f"({pct_zero_both:.2f}% of all users)\n"
                    )

                    f.write(
                        f"Cold-start zero-hit users: "
                        f"{n_zero_both_cold}/{n_cold} "
                        f"({pct_zero_both_cold:.2f}% of cold-start users)\n"
                    )

                    f.write(
                        f"Non-cold-start zero-hit users: "
                        f"{n_zero_both_non_cold}/{n_non_cold} "
                        f"({pct_zero_both_non_cold:.2f}% of non-cold-start users)\n"
                    )


def plot_candidate_pooling_forgotten_users() -> None:
    """Scatter plot of hit-count difference between the oracle candidate-pooling ensemble and popularity.

    The candidate-pooling ensemble uses an oracle-style aggregation:
    Hit_{oracle}(u) = max_m Hit_m(u) across all recommenders.
    """

    for ds_name in DATASETS:
        for model_seed in MODEL_SEEDS:
            candidate_hit_counts = None
            pop_hit_counts = None
            user_history_lengths = None
            user_ids = None

            for split_id in SPLIT_SEEDS:
                test_path = (
                    Path(CHECKPOINT_DIR)
                    / ds_name
                    / f"model_{model_seed}"
                    / "data"
                    / f"split_{split_id}"
                    / "test_data.pkl"
                )

                with open(test_path, "rb") as f:
                    test_in = pickle.load(f)
                    test_out = pickle.load(f)

                # compute per-user oracle hit by taking the maximum hit indicator across recommenders
                candidate_hits = None

                for model_name in ALGORITHM_CONFIGS:
                    if model_name.lower() == "popularity":
                        continue

                    pred_path = (
                        Path(CHECKPOINT_DIR)
                        / ds_name
                        / f"model_{model_seed}"
                        / "algorithms"
                        / model_name
                        / f"split_{split_id}"
                        / "predictions.pkl"
                    )

                    with open(pred_path, "rb") as f:
                        model_pred = pickle.load(f)

                    hit_metric = HitK(K=K)
                    hit_metric.calculate(test_out.binary_values, model_pred)
                    model_hits = (
                        hit_metric.results.groupby("user_id")["score"]
                        .any()
                        .astype(int)
                        .reset_index()
                    )

                    if candidate_hits is None:
                        candidate_hits = model_hits["score"].values
                    else:
                        candidate_hits = np.maximum(candidate_hits, model_hits["score"].values)

                pop_path = (
                    Path(CHECKPOINT_DIR)
                    / ds_name
                    / f"model_{model_seed}"
                    / "algorithms"
                    / "Popularity"
                    / f"split_{split_id}"
                    / "predictions.pkl"
                )
                with open(pop_path, "rb") as f:
                    pop_pred = pickle.load(f)

                hit_metric = HitK(K=K)
                hit_metric.calculate(test_out.binary_values, pop_pred)
                pop_hits = (
                    hit_metric.results.groupby("user_id")["score"]
                    .any()
                    .astype(int)
                    .reset_index()
                )

                if user_ids is None:
                    user_ids = pop_hits["user_id"].values
                    candidate_hit_counts = np.zeros(len(user_ids), dtype=int)
                    pop_hit_counts = np.zeros(len(user_ids), dtype=int)
                    active_users = sorted(test_in.active_users)
                    hist = test_in.values[active_users].getnnz(axis=1)
                    user_history_lengths = pd.Series(hist, index=active_users)

                candidate_hit_counts += candidate_hits
                pop_hit_counts += pop_hits["score"].values

            df = pd.DataFrame(
                {
                    "user_id": user_ids,
                    "history_length": user_history_lengths.reindex(user_ids).values,
                    "candidate_hits": candidate_hit_counts,
                    "pop_hits": pop_hit_counts,
                }
            )
            df["hit_difference"] = df["candidate_hits"] - df["pop_hits"]

            cold_start_threshold = get_cold_start(ds_name)
            cold_start_mask = df["history_length"] <= cold_start_threshold
            
            worse_mask = (
                (df["hit_difference"] < 0)
                | (
                    (df["hit_difference"] == 0)
                    & (df["candidate_hits"] == 0)
                    & (df["pop_hits"] == 0)
                )
            )

            forgotten_mask = (~cold_start_mask) & worse_mask
            normal_mask = (~cold_start_mask) & ~worse_mask

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.scatter(
                df.loc[cold_start_mask, "history_length"],
                df.loc[cold_start_mask, "hit_difference"],
                color="grey",
                marker="s",
                edgecolors="none",
                alpha=0.7,
                s=12,
                label=f"Cold-start users (n={cold_start_mask.sum()})",
            )

            ax.scatter(
                df.loc[normal_mask, "history_length"],
                df.loc[normal_mask, "hit_difference"],
                color="grey",
                marker="o",
                edgecolors="none",
                alpha=0.7,
                s=12,
                label=f"Regular users (n={normal_mask.sum()})",
            )

            if forgotten_mask.any():
                ax.scatter(
                    df.loc[forgotten_mask, "history_length"],
                    df.loc[forgotten_mask, "hit_difference"],
                    color="orange",
                    marker="o",
                    edgecolors="none",
                    alpha=0.7,
                    s=12,
                    label=f"Forgotten users (n={forgotten_mask.sum()})",
                )

            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(cold_start_threshold, color="gray", linestyle=":", linewidth=1)
            ax.set_ylim(-20.5, 20.5)
            ax.set_xlabel("History Length")
            ax.set_ylabel("Oracle Candidate-Pooling Hit Difference")
            ax.legend(loc="upper right")

            plt.savefig(
                _result_dir(ds_name, model_seed, "CandidatePooling") / "candidate_pooling_forgotten_users.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

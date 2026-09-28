# Forgotten Users

This repository implements the experimental framework used to study “forgotten users” in recommender systems: users who are not in the cold-start regime, but still perform worse under a learned recommender than under a simple popularity baseline. The project is designed to support a reproducible research workflow, from raw data preprocessing to model training, evaluation, and diagnostic visualization.

**Paper:** materials -> forgotten_users(paper).pdf
**Poster:** materials -> forgotten_users(poster).png
**Presentation Slides:** materials -> forgotten_users(slides).pdf
**Code:** src

## Research goal

The central question is whether recommender systems systematically forget some users even after those users have accumulated enough history to be considered established. To study this, the codebase builds a controlled evaluation pipeline that compares multiple recommenders against a popularity baseline and quantifies users who are forgotten by the model.

## What the project does

The pipeline covers the full experimental lifecycle:

1. Data loading and preprocessing for multiple recommendation datasets.
2. Construction of train/validation/test splits with strong-generalization logic.
3. Training and hyperparameter selection for several recommender models.
4. Evaluation with ranking metrics such as Hit@K and NDCG@K.
5. Analysis and visualization of forgotten-user behavior through plots and summary statistics.

## Supported datasets

The repository is configured for several public and research-friendly datasets, including:

- MovieLens1M
- MovieLens100K
- CiteULike
- Netflix
- Prime Pantry

Each dataset can be configured with dataset-specific preprocessing and cold-start thresholds in the YAML configuration.

## Methodology

### Preprocessing

Datasets are loaded through RecPack and filtered using configurable preprocessing steps such as:

- minimum rating thresholds
- minimum interactions per user
- minimum interactions per item
- deduplication

### Splits and evaluation protocol

The project uses a strong-generalization split design so that evaluation focuses on users who are not seen during training. Held-out interactions are created for each split, and predictions are compared against both the model and a popularity baseline.

### Models

The repository evaluates:

- Popularity baseline
- ItemKNN
- EASE
- StrongGenSVD, a custom SVD-based implementation adapted for strong generalization via fold-in prediction

### Diagnostics

The analysis module produces:

- scatter plots of hit-rate differences versus history length
- UMAP visualizations of user representations
- candidate-pooling analyses
- summary statistics reporting how many users are forgotten relative to the popularity baseline

## Repository structure

- [config/config.yaml](config/config.yaml): experiment configuration, datasets, seeds, algorithm grids, and evaluation settings.
- [src/main.py](src/main.py): orchestration of the end-to-end experiment pipeline.
- [src/dataset.py](src/dataset.py): dataset registry and preprocessing logic.
- [src/splits.py](src/splits.py): split generation and prediction utilities.
- [src/StrongGenSVD.py](src/StrongGenSVD.py): custom strong-generalization SVD model.
- [src/plots.py](src/plots.py): plotting and statistical reporting routines.
- [data/raw](data/raw): raw dataset files.
- [data/processed](data/processed): processed splits and trained model artifacts.
- [results](results): generated plots and experiment summaries.

## Setup

The project depends on Python and a small set of scientific libraries.

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

From the repository root, run:

```bash
python src/main.py
```

This will:

- build the requested datasets and splits
- train the configured models
- generate evaluation artifacts and plots
- write outputs to the processed-data and results directories

## Output artifacts

For each dataset and model configuration, the pipeline writes:

- prediction files and split objects under [data/processed](data/processed)
- figures and summary files under [results](results)

Representative outputs include files such as:

- forgotten user scatter plots
- UMAP embeddings
- candidate-pooling visualizations
- per-model statistics reports

## Notes on the current results

The saved outputs in [results](results) already show that forgotten users appear across datasets and models, with the effect varying by dataset and recommender family. For example, the repository currently contains results indicating substantial forgotten-user rates for ItemKNN on MovieLens1M and Netflix, while other models show smaller but still measurable effects.

This project is intended to be both reproducible and paper-friendly: it combines a clear experimental protocol with interpretable diagnostic outputs that are suitable for analysis and presentation.


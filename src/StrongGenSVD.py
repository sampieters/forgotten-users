# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix, lil_matrix, diags
import sklearn.decomposition

from recpack.algorithms.base import Algorithm


class StrongGenFactorizationAlgorithm(Algorithm):
    """Base class for factorization algorithms supporting Strong Generalization (fold-in).

    Standard factorization models retain a fixed `user_embedding_` fitted during `_fit`.
    In Strong Generalization, test users are unseen during training and have 0-rows in
    the training matrix.

    This base class overrides `_predict` to fold-in active users dynamically at prediction time:
    their history vector `X` is projected into the latent user space via `self.model_.transform()`,
    and scores are computed as `user_embedding_test @ self.item_embedding_`.

    :param num_components: The dimension of the feature matrices. Defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_components: int = 100):
        super().__init__()
        self.num_components = num_components

    def _check_fit_complete(self):
        super()._check_fit_complete()
        assert hasattr(self, "model_")
        assert hasattr(self, "item_embedding_")
        assert self.item_embedding_.shape[0] == self.num_components

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for active users in the interaction matrix X.

        For active users (rows with non-zero entries in X), fold-in user embeddings are
        computed by projecting their interaction vectors using `self.model_.transform()`.
        Scores are calculated by multiplying fold-in user embeddings with `self.item_embedding_`.

        :param X: binary interaction matrix (users x items).
        :type X: csr_matrix
        :return: matrix with predicted scores for each user.
        :rtype: csr_matrix
        """
        assert X.shape[1] == self.item_embedding_.shape[1]

        active_users = list(set(X.nonzero()[0]))
        if not active_users:
            return csr_matrix(X.shape)

        # Extract fold-in interactions for active users
        X_active = X[active_users]

        # Fold-in: project active user interaction vectors into latent user space
        user_emb_active = self.model_.transform(X_active)

        # Predict scores by multiplying active user embeddings with item embeddings
        scores_active = user_emb_active @ self.item_embedding_

        # Populate result matrix matching full shape X.shape
        scores = lil_matrix(X.shape)
        scores[active_users] = scores_active
        return scores.tocsr()


class StrongGenSVD(StrongGenFactorizationAlgorithm):
    """Singular Value Decomposition adapted for Strong Generalization via Fold-In.

    In Strong Generalization, test users are not seen during model fitting.
    During prediction (`_predict`), fold-in user embeddings are dynamically computed
    by projecting each test user's history vector onto the learned SVD components.

    The SVD is computed using `TruncatedSVD <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_ from sklearn.

    :param num_components: The size of the embeddings. Defaults to 100.
    :type num_components: int
    :param seed: The seed for the random state to allow for comparison, defaults to None
    :type seed: int, optional
    """

    def __init__(self, num_components: int = 100, seed: Optional[int] = None):
        super().__init__(num_components=num_components)
        self.seed = seed

    def _fit(self, X: csr_matrix):
        model = sklearn.decomposition.TruncatedSVD(
            n_components=self.num_components,
            n_iter=7,
            random_state=self.seed
        )
        self.user_embedding_ = model.fit_transform(X)

        V = model.components_
        sigma = diags(model.singular_values_)
        self.item_embedding_ = sigma @ V
        self.model_ = model

        # Post conditions
        assert self.user_embedding_.shape == (X.shape[0], self.num_components)
        assert self.item_embedding_.shape == (self.num_components, X.shape[1])
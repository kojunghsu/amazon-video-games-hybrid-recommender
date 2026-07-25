from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class HybridRecommender:
    """Blend collaborative affinity with a metadata-derived user profile."""

    def __init__(
        self,
        collaborative_model,
        item_features: sparse.csr_matrix,
        history: dict[int, set[int]],
        alpha: float = 0.8,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.collaborative_model = collaborative_model
        self.item_features = item_features.tocsr()
        self.history = history
        self.alpha = alpha
        self._profiles: dict[int, sparse.csr_matrix] = {}

    def _profile(self, user_index: int) -> sparse.csr_matrix:
        if user_index not in self._profiles:
            seen = sorted(self.history.get(user_index, set()))
            if not seen:
                profile = sparse.csr_matrix((1, self.item_features.shape[1]))
            else:
                profile = normalize(
                    self.item_features[seen].mean(axis=0), norm="l2"
                )
                profile = sparse.csr_matrix(profile)
            self._profiles[user_index] = profile
        return self._profiles[user_index]

    def score_pairs(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        collaborative = self.collaborative_model.score_pairs(
            user_indices, item_indices, batch_size
        )
        content = np.zeros(len(item_indices), dtype=np.float32)
        for user in np.unique(user_indices):
            positions = np.flatnonzero(user_indices == user)
            profile = self._profile(int(user))
            content[positions] = (
                self.item_features[item_indices[positions]] @ profile.T
            ).toarray().ravel()
        collaborative = np.tanh(collaborative).astype(np.float32)
        return self.alpha * collaborative + (1 - self.alpha) * content

    def recommend(
        self, user_index: int, history: set[int], k: int
    ) -> np.ndarray:
        item_indices = np.arange(self.item_features.shape[0], dtype=np.int64)
        user_indices = np.full(len(item_indices), user_index, dtype=np.int64)
        scores = self.score_pairs(user_indices, item_indices, batch_size=65536)
        if history:
            scores[np.fromiter(history, dtype=np.int64)] = -np.inf
        candidate_count = min(k, len(scores))
        ranked = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        return ranked[np.argsort(-scores[ranked])]


def build_content_features(
    content_by_item: list[str],
    max_features: int = 30_000,
    minimum_document_frequency: int = 2,
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=minimum_document_frequency,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    features = vectorizer.fit_transform(content_by_item)
    return vectorizer, normalize(features, norm="l2").tocsr()


def similar_items(
    item_index: int,
    item_features: sparse.csr_matrix,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    scores = (item_features @ item_features[item_index].T).toarray().ravel()
    scores[item_index] = -np.inf
    candidate_count = min(k, len(scores) - 1)
    ranked = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
    ranked = ranked[np.argsort(-scores[ranked])]
    return ranked, scores[ranked]

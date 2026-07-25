from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityRecommender:
    def fit(self, train: pd.DataFrame, number_of_items: int) -> "PopularityRecommender":
        counts = train["item_index"].value_counts()
        self.scores = np.zeros(number_of_items, dtype=np.float32)
        self.scores[counts.index.to_numpy(dtype=int)] = counts.to_numpy(dtype=float)
        return self

    def score_items(self, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        del user_index
        return self.scores[item_indices]

    def score_pairs(
        self, user_indices: np.ndarray, item_indices: np.ndarray, batch_size: int
    ) -> np.ndarray:
        del user_indices, batch_size
        return self.scores[item_indices]

    def recommend(
        self, user_index: int, history: set[int], k: int
    ) -> np.ndarray:
        del user_index
        scores = self.scores.copy()
        if history:
            scores[np.fromiter(history, dtype=int)] = -np.inf
        return np.argpartition(-scores, min(k, len(scores) - 1))[:k][
            np.argsort(-scores[np.argpartition(-scores, min(k, len(scores) - 1))[:k]])
        ]

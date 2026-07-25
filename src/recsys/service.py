from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .bpr import BPRMatrixFactorization
from .hybrid import HybridRecommender, similar_items


class RecommendationService:
    def __init__(self, artifact_path: str | Path) -> None:
        checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
        self.user_to_index = checkpoint["user_to_index"]
        self.item_to_index = checkpoint["item_to_index"]
        self.index_to_item = {
            index: item_id for item_id, index in self.item_to_index.items()
        }
        self.history = checkpoint["history"]
        self.popularity_scores = np.asarray(checkpoint["popularity_scores"])
        self.catalog = checkpoint.get("catalog", {})
        self.item_features = checkpoint.get("item_features")
        config = checkpoint["config"]
        self.model = BPRMatrixFactorization(
            len(self.user_to_index),
            len(self.item_to_index),
            int(config["embedding_dim"]),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        if self.item_features is not None:
            self.ranker = HybridRecommender(
                self.model,
                self.item_features,
                self.history,
                float(checkpoint["hybrid_alpha"]),
            )
        else:
            self.ranker = self.model

    def _item_payload(
        self, item_index: int, rank: int, score: float | None = None
    ) -> dict:
        item_id = self.index_to_item[int(item_index)]
        payload = {"rank": rank, "item_id": item_id}
        payload.update(self.catalog.get(item_id, {}))
        if score is not None:
            payload["similarity"] = round(float(score), 4)
        return payload

    def recommend(self, user_id: str, k: int = 10) -> dict:
        if k < 1 or k > 100:
            raise ValueError("k must be between 1 and 100")

        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            ranked = np.argsort(-self.popularity_scores)[:k]
            strategy = "popularity_cold_start"
        else:
            ranked = self.ranker.recommend(
                user_index, self.history.get(user_index, set()), k
            )
            strategy = (
                "personalized_hybrid"
                if self.item_features is not None
                else "personalized_bpr"
            )

        return {
            "user_id": user_id,
            "strategy": strategy,
            "recommendations": [
                self._item_payload(int(item_index), rank)
                for rank, item_index in enumerate(ranked, start=1)
            ],
        }

    def similar(self, item_id: str, k: int = 10) -> dict:
        if self.item_features is None:
            raise RuntimeError("Content features are unavailable")
        item_index = self.item_to_index.get(item_id)
        if item_index is None:
            raise KeyError(item_id)
        ranked, scores = similar_items(item_index, self.item_features, k)
        return {
            "seed_item": self.catalog.get(item_id, {"item_id": item_id}),
            "strategy": "metadata_content_similarity",
            "recommendations": [
                self._item_payload(index, rank, score)
                for rank, (index, score) in enumerate(
                    zip(ranked, scores), start=1
                )
            ],
        }

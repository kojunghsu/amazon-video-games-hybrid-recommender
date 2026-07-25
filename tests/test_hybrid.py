import numpy as np
from scipy import sparse

from recsys.hybrid import HybridRecommender, build_content_features, similar_items


class ZeroCollaborativeModel:
    def score_pairs(self, users, items, batch_size):
        del users, batch_size
        return np.zeros(len(items), dtype=np.float32)


def test_content_similarity_returns_related_item() -> None:
    _, features = build_content_features(
        ["space strategy galaxy", "space galaxy adventure", "football sports"],
        minimum_document_frequency=1,
    )
    ranked, _ = similar_items(0, features, k=1)
    assert ranked.tolist() == [1]


def test_hybrid_uses_history_as_content_profile() -> None:
    features = sparse.csr_matrix(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    )
    hybrid = HybridRecommender(
        ZeroCollaborativeModel(), features, {0: {0}}, alpha=0.0
    )
    ranked = hybrid.recommend(0, {0}, k=1)
    assert ranked.tolist() == [1]

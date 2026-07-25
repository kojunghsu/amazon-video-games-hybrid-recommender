import numpy as np

from recsys.metrics import (
    catalog_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    reciprocal_rank,
)


def test_single_relevant_item_metrics() -> None:
    ranked = np.array([9, 4, 2, 7])
    assert hit_rate_at_k(ranked, 4, 1) == 0
    assert hit_rate_at_k(ranked, 4, 2) == 1
    assert ndcg_at_k(ranked, 4, 2) > 0
    assert reciprocal_rank(ranked, 4) == 0.5


def test_catalog_coverage() -> None:
    recommendations = [np.array([1, 2]), np.array([2, 3])]
    assert catalog_coverage(recommendations, 5) == 0.6


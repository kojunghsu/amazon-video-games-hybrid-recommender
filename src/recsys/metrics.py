from __future__ import annotations

import math

import numpy as np


def hit_rate_at_k(ranked_items: np.ndarray, target_item: int, k: int) -> float:
    return float(target_item in ranked_items[:k])


def recall_at_k(ranked_items: np.ndarray, target_item: int, k: int) -> float:
    return hit_rate_at_k(ranked_items, target_item, k)


def ndcg_at_k(ranked_items: np.ndarray, target_item: int, k: int) -> float:
    matches = np.flatnonzero(ranked_items[:k] == target_item)
    return 0.0 if len(matches) == 0 else 1.0 / math.log2(int(matches[0]) + 2)


def reciprocal_rank(ranked_items: np.ndarray, target_item: int) -> float:
    matches = np.flatnonzero(ranked_items == target_item)
    return 0.0 if len(matches) == 0 else 1.0 / (int(matches[0]) + 1)


def catalog_coverage(recommendations: list[np.ndarray], item_count: int) -> float:
    if not recommendations or item_count <= 0:
        return 0.0
    return len(set(np.concatenate(recommendations).tolist())) / item_count


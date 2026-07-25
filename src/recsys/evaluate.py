from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .metrics import catalog_coverage, hit_rate_at_k, ndcg_at_k, reciprocal_rank


def sampled_candidates(
    target: int,
    history: set[int],
    number_of_items: int,
    negative_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    excluded = history | {target}
    negatives: set[int] = set()
    while len(negatives) < min(negative_count, number_of_items - len(excluded)):
        candidate = int(rng.integers(number_of_items))
        if candidate not in excluded:
            negatives.add(candidate)
    return np.asarray([target, *sorted(negatives)], dtype=np.int64)


def evaluate_ranking(
    model,
    evaluation: pd.DataFrame,
    history: dict[int, set[int]],
    number_of_items: int,
    ks: tuple[int, ...] = (5, 10, 20),
    negative_count: int = 100,
    seed: int = 42,
    maximum_users: int | None = None,
    batch_size: int = 65536,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    totals: defaultdict[str, float] = defaultdict(float)
    recommendations: list[np.ndarray] = []

    if maximum_users and len(evaluation) > maximum_users:
        evaluation = evaluation.sample(n=maximum_users, random_state=seed)

    candidate_rows = []
    users = []
    targets = []
    for row in evaluation.itertuples(index=False):
        user = int(row.user_index)
        target = int(row.item_index)
        candidates = sampled_candidates(
            target, history.get(user, set()), number_of_items, negative_count, rng
        )
        users.append(user)
        targets.append(target)
        candidate_rows.append(candidates)

    candidate_matrix = np.stack(candidate_rows)
    user_matrix = np.repeat(np.asarray(users)[:, None], candidate_matrix.shape[1], axis=1)
    flat_scores = model.score_pairs(
        user_matrix.ravel(), candidate_matrix.ravel(), batch_size
    )
    score_matrix = flat_scores.reshape(candidate_matrix.shape)

    for candidates, scores, target in zip(candidate_matrix, score_matrix, targets):
        ranked = candidates[np.argsort(-scores)]
        recommendations.append(ranked[: max(ks)])
        totals["MRR"] += reciprocal_rank(ranked, target)
        for k in ks:
            totals[f"HitRate@{k}"] += hit_rate_at_k(ranked, target, k)
            totals[f"NDCG@{k}"] += ndcg_at_k(ranked, target, k)

    count = max(len(targets), 1)
    results = {metric: value / count for metric, value in totals.items()}
    results[f"Coverage@{max(ks)}"] = catalog_coverage(
        recommendations, number_of_items
    )
    results["evaluated_users"] = float(len(targets))
    return results

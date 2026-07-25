from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"user_id", "parent_asin", "rating", "timestamp"}


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    history: dict[int, set[int]]
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_item: dict[int, str]


def load_interactions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame[list(REQUIRED_COLUMNS)].rename(columns={"parent_asin": "item_id"})
    frame = frame.dropna().copy()
    frame["rating"] = frame["rating"].astype("float32")
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame = frame.sort_values(["user_id", "timestamp", "item_id"])
    frame = frame.drop_duplicates(["user_id", "item_id"], keep="last")
    return frame.reset_index(drop=True)


def build_leave_two_out_split(
    interactions: pd.DataFrame,
    positive_threshold: float = 4.0,
    minimum_positives: int = 3,
) -> TemporalSplit:
    """Create leakage-safe per-user train/validation/test ranking splits.

    The latest positive interaction is test, the second latest is validation,
    and all earlier positives are training history. Users with too little
    positive history remain useful for popularity statistics but are excluded
    from personalized offline evaluation.
    """
    positives = interactions[interactions["rating"] >= positive_threshold].copy()
    positives = positives.sort_values(["user_id", "timestamp", "item_id"])
    eligible = positives.groupby("user_id").size()
    eligible_users = eligible[eligible >= minimum_positives].index
    positives = positives[positives["user_id"].isin(eligible_users)].copy()

    user_ids = sorted(positives["user_id"].unique())
    item_ids = sorted(interactions["item_id"].unique())
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}

    positives["user_index"] = positives["user_id"].map(user_to_index)
    positives["item_index"] = positives["item_id"].map(item_to_index)
    reverse_rank = positives.groupby("user_id").cumcount(ascending=False)

    test = positives[reverse_rank == 0].copy()
    validation = positives[reverse_rank == 1].copy()
    train = positives[reverse_rank >= 2].copy()
    history = (
        train.groupby("user_index")["item_index"]
        .agg(lambda values: set(values.astype(int)))
        .to_dict()
    )
    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        history=history,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_item={index: value for value, index in item_to_index.items()},
    )


class BPRTrainingDataset:
    """Generate negative samples without materializing the user-item matrix."""

    def __init__(
        self,
        train: pd.DataFrame,
        history: dict[int, set[int]],
        number_of_items: int,
        negatives_per_positive: int,
        seed: int,
    ) -> None:
        self.users = train["user_index"].to_numpy(dtype=np.int64)
        self.positives = train["item_index"].to_numpy(dtype=np.int64)
        self.history = history
        self.number_of_items = number_of_items
        self.negatives_per_positive = negatives_per_positive
        self.seed = seed

    def sample_epoch(self, epoch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + epoch)
        users = np.repeat(self.users, self.negatives_per_positive)
        positives = np.repeat(self.positives, self.negatives_per_positive)
        negatives = np.empty_like(users)
        for index, user in enumerate(users):
            candidate = int(rng.integers(self.number_of_items))
            while candidate in self.history[int(user)]:
                candidate = int(rng.integers(self.number_of_items))
            negatives[index] = candidate
        return users, positives, negatives


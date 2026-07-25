import pandas as pd

from recsys.data import build_leave_two_out_split


def test_leave_two_out_is_temporal_and_leakage_safe() -> None:
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3", "i4"],
            "rating": [5.0, 4.0, 5.0, 4.0],
            "timestamp": [1, 2, 3, 4],
        }
    )
    split = build_leave_two_out_split(frame)
    assert split.train.iloc[-1]["item_id"] == "i2"
    assert split.validation.iloc[0]["item_id"] == "i3"
    assert split.test.iloc[0]["item_id"] == "i4"


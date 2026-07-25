from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


METADATA_FIELDS = {
    "parent_asin",
    "title",
    "main_category",
    "categories",
    "store",
    "features",
    "description",
    "price",
    "average_rating",
    "rating_number",
    "images",
}


def _as_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(part) for part in value if part)
    return "" if value is None else str(value)


def load_metadata(
    path: str | Path,
    item_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Stream JSONL metadata and retain only the recommendation catalog."""
    selected = set(item_ids) if item_ids is not None else None
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on metadata line {line_number}"
                ) from error
            item_id = record.get("parent_asin")
            if not item_id or (selected is not None and item_id not in selected):
                continue
            records.append({key: record.get(key) for key in METADATA_FIELDS})

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No metadata matched the recommendation catalog")
    frame = frame.drop_duplicates("parent_asin", keep="last")
    frame["content_text"] = frame.apply(
        lambda row: " ".join(
            part
            for part in (
                _as_text(row.get("title")),
                _as_text(row.get("main_category")),
                _as_text(row.get("categories")),
                _as_text(row.get("store")),
                _as_text(row.get("features")),
                _as_text(row.get("description")),
            )
            if part
        ),
        axis=1,
    )
    return frame.reset_index(drop=True)


def metadata_coverage(metadata: pd.DataFrame, item_ids: Iterable[str]) -> float:
    catalog = set(item_ids)
    matched = set(metadata["parent_asin"]).intersection(catalog)
    return len(matched) / max(len(catalog), 1)


def public_item_record(record: dict) -> dict:
    images = record.get("images")
    image_url = None
    if isinstance(images, list) and images:
        image_url = images[0].get("large") or images[0].get("thumb")
    payload = {
        "item_id": record.get("parent_asin"),
        "title": record.get("title"),
        "main_category": record.get("main_category"),
        "store": record.get("store"),
        "price": record.get("price"),
        "average_rating": record.get("average_rating"),
        "rating_number": record.get("rating_number"),
        "image_url": image_url,
    }
    return {
        key: None if not isinstance(value, (list, dict)) and pd.isna(value) else value
        for key, value in payload.items()
    }

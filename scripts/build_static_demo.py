from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


SEGMENTS = {
    "popular": (),
    "nintendo": ("nintendo", "switch", "mario", "zelda", "pokemon"),
    "console": ("playstation", "ps4", "ps5", "xbox"),
    "accessories": (
        "controller",
        "headset",
        "keyboard",
        "mouse",
        "charging",
        "adapter",
    ),
}


def positive_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if float(row["rating"]) >= 4:
                counts[row["parent_asin"]] += 1
    return counts


def public_record(record: dict, interactions: int) -> dict:
    images = record.get("images") or []
    image_url = None
    if images and isinstance(images[0], dict):
        image_url = images[0].get("large") or images[0].get("thumb")
    return {
        "item_id": record["parent_asin"],
        "title": record.get("title"),
        "store": record.get("store"),
        "category": record.get("main_category"),
        "price": record.get("price"),
        "average_rating": record.get("average_rating"),
        "rating_number": record.get("rating_number"),
        "image_url": image_url,
        "positive_interactions": interactions,
    }


def build_demo(
    interactions_path: Path, metadata_path: Path, output_path: Path
) -> None:
    counts = positive_counts(interactions_path)
    candidates: dict[str, list[tuple[int, dict]]] = {
        segment: [] for segment in SEGMENTS
    }
    with metadata_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            item_id = record.get("parent_asin")
            interaction_count = counts.get(item_id, 0)
            if not item_id or interaction_count == 0:
                continue
            searchable = " ".join(
                str(record.get(field) or "")
                for field in ("title", "main_category", "categories", "store")
            ).lower()
            payload = public_record(record, interaction_count)
            candidates["popular"].append((interaction_count, payload))
            for segment, keywords in SEGMENTS.items():
                if segment != "popular" and any(
                    keyword in searchable for keyword in keywords
                ):
                    candidates[segment].append((interaction_count, payload))

    recommendations = {}
    for segment, rows in candidates.items():
        ranked = sorted(rows, key=lambda row: row[0], reverse=True)[:8]
        recommendations[segment] = [payload for _, payload in ranked]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "generated_from": "Amazon Reviews 2023 — Video Games",
                "ranking": "positive-interaction popularity within segment",
                "recommendations": recommendations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/data.json"))
    args = parser.parse_args()
    build_demo(args.interactions, args.metadata, args.output)


if __name__ == "__main__":
    main()

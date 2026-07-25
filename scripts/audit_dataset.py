import csv
import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone


if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/audit_dataset.py /path/to/Video_Games.csv.gz")

DATA_PATH = sys.argv[1]


def quantiles(values, probabilities):
    ordered = sorted(values)
    last = len(ordered) - 1
    return {
        str(probability): ordered[round(probability * last)]
        for probability in probabilities
    }


users = Counter()
items = Counter()
ratings = Counter()
user_item_pairs = set()
rows = []
duplicate_pairs = 0

with gzip.open(DATA_PATH, "rt", newline="") as source:
    reader = csv.DictReader(source)
    for row in reader:
        user_id = row["user_id"]
        item_id = row["parent_asin"]
        rating = float(row["rating"])
        timestamp = int(row["timestamp"])
        pair = (user_id, item_id)
        duplicate_pairs += pair in user_item_pairs
        user_item_pairs.add(pair)
        users[user_id] += 1
        items[item_id] += 1
        ratings[rating] += 1
        rows.append((timestamp, user_id, item_id))

rows.sort()
split_index = int(len(rows) * 0.8)
train_rows = rows[:split_index]
test_rows = rows[split_index:]
train_users = {row[1] for row in train_rows}
train_items = {row[2] for row in train_rows}
new_user_count = sum(row[1] not in train_users for row in test_rows)
new_item_count = sum(row[2] not in train_items for row in test_rows)
cold_count = sum(
    row[1] not in train_users or row[2] not in train_items for row in test_rows
)

result = {
    "rows": len(rows),
    "users": len(users),
    "items": len(items),
    "duplicate_user_item_pairs": duplicate_pairs,
    "date_min": datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc).isoformat(),
    "date_max": datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat(),
    "ratings": dict(sorted(ratings.items())),
    "user_interactions_quantiles": quantiles(
        users.values(), [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    ),
    "item_interactions_quantiles": quantiles(
        items.values(), [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    ),
    "density": len(rows) / (len(users) * len(items)),
    "temporal_80_20_train": len(train_rows),
    "temporal_80_20_test": len(test_rows),
    "cold_test_rate": cold_count / len(test_rows),
    "new_user_rate": new_user_count / len(test_rows),
    "new_item_rate": new_item_count / len(test_rows),
}

print(json.dumps(result, indent=2))

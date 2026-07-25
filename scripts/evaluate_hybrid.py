from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from recsys.bpr import BPRMatrixFactorization
from recsys.data import build_leave_two_out_split, load_interactions
from recsys.evaluate import evaluate_ranking
from recsys.hybrid import HybridRecommender, build_content_features
from recsys.metadata import load_metadata, metadata_coverage, public_item_record


def main(args: argparse.Namespace) -> None:
    artifact_path = Path(args.model)
    checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
    interactions = load_interactions(args.data)
    split = build_leave_two_out_split(
        interactions,
        positive_threshold=checkpoint["config"]["positive_rating_threshold"],
        minimum_positives=checkpoint["config"]["minimum_positive_interactions"],
    )
    model = BPRMatrixFactorization(
        len(split.user_to_index),
        len(split.item_to_index),
        int(checkpoint["config"]["embedding_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metadata = load_metadata(args.metadata, split.item_to_index)
    by_id = metadata.set_index("parent_asin").to_dict("index")
    content = [
        by_id.get(split.index_to_item[index], {}).get("content_text", "")
        for index in range(len(split.item_to_index))
    ]
    vectorizer, features = build_content_features(content)

    config = checkpoint["config"]
    evaluation_options = dict(
        history=split.history,
        number_of_items=len(split.item_to_index),
        ks=tuple(config["top_k"]),
        negative_count=config["evaluation_negatives"],
        seed=config["seed"],
        maximum_users=args.maximum_users,
        batch_size=config["evaluation_batch_size"],
    )
    validation_scores: dict[str, dict[str, float]] = {}
    for alpha in args.alphas:
        hybrid = HybridRecommender(model, features, split.history, alpha)
        validation_scores[str(alpha)] = evaluate_ranking(
            hybrid, split.validation, **evaluation_options
        )
    best_alpha = max(
        args.alphas,
        key=lambda value: validation_scores[str(value)]["NDCG@10"],
    )
    hybrid = HybridRecommender(model, features, split.history, best_alpha)
    test_metrics = evaluate_ranking(hybrid, split.test, **evaluation_options)

    catalog = {
        item_id: public_item_record({"parent_asin": item_id, **record})
        for item_id, record in by_id.items()
    }
    checkpoint["content_vectorizer"] = vectorizer
    checkpoint["item_features"] = features
    checkpoint["catalog"] = catalog
    checkpoint["hybrid_alpha"] = best_alpha
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    result = {
        "metadata": {
            "matched_items": len(metadata),
            "catalog_items": len(split.item_to_index),
            "coverage": metadata_coverage(metadata, split.item_to_index),
            "feature_count": features.shape[1],
        },
        "validation_alpha_search": validation_scores,
        "selected_alpha": best_alpha,
        "Hybrid": test_metrics,
    }
    (output.parent / "hybrid_metrics.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--model", default="artifacts/bpr_model.pt")
    parser.add_argument("--output", default="artifacts/hybrid_model.pt")
    parser.add_argument("--maximum-users", type=int, default=10_000)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9]
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

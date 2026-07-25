from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .baselines import PopularityRecommender
from .bpr import BPRMatrixFactorization
from .data import BPRTrainingDataset, build_leave_two_out_split, load_interactions
from .evaluate import evaluate_ranking


def train(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.config).read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    interactions = load_interactions(args.data)
    split = build_leave_two_out_split(
        interactions,
        positive_threshold=config["positive_rating_threshold"],
        minimum_positives=config["minimum_positive_interactions"],
    )
    item_count = len(split.item_to_index)
    user_count = len(split.user_to_index)
    ks = tuple(config["top_k"])

    popularity = PopularityRecommender().fit(split.train, item_count)
    popularity_metrics = evaluate_ranking(
        popularity,
        split.test,
        split.history,
        item_count,
        ks=ks,
        negative_count=config["evaluation_negatives"],
        seed=seed,
        maximum_users=config["maximum_evaluation_users"],
        batch_size=config["evaluation_batch_size"],
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = BPRMatrixFactorization(
        user_count, item_count, config["embedding_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    sampler = BPRTrainingDataset(
        split.train,
        split.history,
        item_count,
        config["negatives_per_positive"],
        seed,
    )

    best_ndcg = -1.0
    patience = 0
    best_state = None
    batch_size = int(config["batch_size"])
    for epoch in range(int(config["epochs"])):
        users, positives, negatives = sampler.sample_epoch(epoch)
        order = np.random.default_rng(seed + epoch).permutation(len(users))
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            user_tensor = torch.as_tensor(users[batch], device=device)
            positive_tensor = torch.as_tensor(positives[batch], device=device)
            negative_tensor = torch.as_tensor(negatives[batch], device=device)
            optimizer.zero_grad()
            loss = model.pairwise_loss(
                user_tensor, positive_tensor, negative_tensor
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_metrics = evaluate_ranking(
            model,
            split.validation,
            split.history,
            item_count,
            ks=ks,
            negative_count=config["evaluation_negatives"],
            seed=seed,
            maximum_users=config["maximum_evaluation_users"],
            batch_size=config["evaluation_batch_size"],
        )
        score = validation_metrics["NDCG@10"]
        print(
            f"epoch={epoch + 1} loss={np.mean(losses):.5f} "
            f"validation_ndcg@10={score:.5f}"
        )
        if score > best_ndcg:
            best_ndcg = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    bpr_metrics = evaluate_ranking(
        model,
        split.test,
        split.history,
        item_count,
        ks=ks,
        negative_count=config["evaluation_negatives"],
        seed=seed,
        maximum_users=config["maximum_evaluation_users"],
        batch_size=config["evaluation_batch_size"],
    )

    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "user_to_index": split.user_to_index,
            "item_to_index": split.item_to_index,
            "history": split.history,
            "popularity_scores": popularity.scores,
            "config": config,
        },
        artifact_dir / "bpr_model.pt",
    )
    results = {"Popularity": popularity_metrics, "BPR-MF": bpr_metrics}
    (artifact_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--artifacts", default="artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

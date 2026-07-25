from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BPRMatrixFactorization(nn.Module):
    """Pairwise-ranking MF optimized for recommendation, not rating RMSE."""

    def __init__(self, users: int, items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(users, embedding_dim)
        self.item_embedding = nn.Embedding(items, embedding_dim)
        self.item_bias = nn.Embedding(items, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        interaction = (
            self.user_embedding(users) * self.item_embedding(items)
        ).sum(dim=1)
        return interaction + self.item_bias(items).squeeze(1)

    def pairwise_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        difference = self(users, positives) - self(users, negatives)
        return -torch.nn.functional.logsigmoid(difference).mean()

    @torch.no_grad()
    def score_items(self, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        users = torch.full(
            (len(item_indices),), user_index, dtype=torch.long, device=device
        )
        items = torch.as_tensor(item_indices, dtype=torch.long, device=device)
        return self(users, items).cpu().numpy()

    @torch.no_grad()
    def score_pairs(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        device = next(self.parameters()).device
        scores = []
        for start in range(0, len(user_indices), batch_size):
            end = start + batch_size
            users = torch.as_tensor(
                user_indices[start:end], dtype=torch.long, device=device
            )
            items = torch.as_tensor(
                item_indices[start:end], dtype=torch.long, device=device
            )
            scores.append(self(users, items).cpu().numpy())
        return np.concatenate(scores)

    @torch.no_grad()
    def recommend(
        self, user_index: int, history: set[int], k: int
    ) -> np.ndarray:
        device = next(self.parameters()).device
        user = self.user_embedding.weight[user_index]
        scores = self.item_embedding.weight @ user + self.item_bias.weight.squeeze(1)
        if history:
            seen = torch.as_tensor(list(history), dtype=torch.long, device=device)
            scores[seen] = -torch.inf
        return torch.topk(scores, k=k).indices.cpu().numpy()

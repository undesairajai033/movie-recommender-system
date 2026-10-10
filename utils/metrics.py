"""Ranking metrics for top-K recommendations.

Precision@K  - of the K movies recommended, what share did the user actually like?
Recall@K     - of the movies the user liked, what share appeared in the top K?
NDCG@K       - like precision, but rewards putting liked movies near the top.
"""

import numpy as np


def precision_at_k(recommended_ids: list, relevant_ids: list, k: int = 10) -> float:
    if not recommended_ids or not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for m in recommended_ids[:k] if m in relevant)
    return hits / k


def recall_at_k(recommended_ids: list, relevant_ids: list, k: int = 10) -> float:
    if not recommended_ids or not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for m in recommended_ids[:k] if m in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended_ids: list, relevant_ids: list, k: int = 10) -> float:
    if not recommended_ids or not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    dcg = sum(1 / np.log2(rank + 2) for rank, m in enumerate(recommended_ids[:k]) if m in relevant)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

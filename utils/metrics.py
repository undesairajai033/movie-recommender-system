# utils/metrics.py
# PURPOSE: Evaluate HOW GOOD our recommendations are
#
# WHY THIS MATTERS FOR YOUR RESUME:
#   Most students just print accuracy. Using Precision@K, Recall@K, NDCG
#   is what separates a student project from a professional one.
#   Mention these in interviews!

import numpy as np
import pandas as pd


def precision_at_k(recommended_ids: list, relevant_ids: list, k: int = 10) -> float:
    """
    Precision@K = "Of the top K recommendations, how many were actually relevant?"

    Example:
        recommended = [1, 2, 3, 4, 5]  (top 5 movies we suggested)
        relevant    = [2, 4, 6, 8]     (movies user actually liked)
        precision@5 = 2/5 = 0.4        (2 out of 5 suggestions were liked)

    Higher = better. Range: 0 to 1.
    """
    if not recommended_ids or not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for movie_id in top_k if movie_id in relevant_set)
    return round(hits / k, 4)


def recall_at_k(recommended_ids: list, relevant_ids: list, k: int = 10) -> float:
    """
    Recall@K = "Of all movies the user actually likes, how many did we find?"

    Example:
        recommended = [1, 2, 3, 4, 5]
        relevant    = [2, 4, 6, 8]
        recall@5    = 2/4 = 0.5   (found 2 out of 4 movies user likes)

    Higher = better. Range: 0 to 1.
    """
    if not recommended_ids or not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for movie_id in top_k if movie_id in relevant_set)
    return round(hits / len(relevant_ids), 4)


def ndcg_at_k(recommended_ids: list, relevant_ids: list, k: int = 10) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain)
    = "Did we put the BEST recommendations at the TOP of the list?"

    Unlike Precision@K, this PENALIZES you if good movies appear at position 8
    instead of position 1. Position matters!

    Higher = better. Range: 0 to 1.
    """
    relevant_set = set(relevant_ids)
    top_k = recommended_ids[:k]

    # DCG: gain is 1 for hit, discounted by log2(rank+1)
    dcg = sum(
        1 / np.log2(rank + 2)  # +2 because rank starts at 0
        for rank, movie_id in enumerate(top_k)
        if movie_id in relevant_set
    )

    # Ideal DCG: what if we put all relevant movies first?
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


def evaluate_model(recommender, ratings_df: pd.DataFrame, k: int = 10, sample_users: int = 50) -> dict:
    """
    Evaluate a recommender model on a sample of users.
    Returns average Precision@K, Recall@K, and NDCG@K.

    This is what you show in your README results table!
    """
    precisions, recalls, ndcgs = [], [], []

    # Sample users for faster evaluation
    user_ids = ratings_df['userId'].unique()
    sample = np.random.choice(user_ids, min(sample_users, len(user_ids)), replace=False)

    for user_id in sample:
        # Split this user's ratings: 80% "known", 20% "test" (ground truth)
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        if len(user_ratings) < 5:
            continue

        test_ratings = user_ratings[user_ratings['rating'] >= 4.0]  # "liked" = rating >= 4
        relevant_ids = test_ratings['movieId'].tolist()

        if not relevant_ids:
            continue

        # Get recommendations
        try:
            recs = recommender.recommend_for_user(user_id, n=k)
            if recs.empty:
                continue
            rec_ids = recs['movieId'].tolist()
        except Exception:
            continue

        precisions.append(precision_at_k(rec_ids, relevant_ids, k))
        recalls.append(recall_at_k(rec_ids, relevant_ids, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant_ids, k))

    return {
        f'Precision@{k}': round(np.mean(precisions), 4) if precisions else 0,
        f'Recall@{k}': round(np.mean(recalls), 4) if recalls else 0,
        f'NDCG@{k}': round(np.mean(ndcgs), 4) if ndcgs else 0,
        'Users evaluated': len(precisions)
    }

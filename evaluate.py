"""Offline evaluation of all recommenders on MovieLens (ml-latest-small).

Method
------
* For every user, 20% of their ratings are hidden as a test set (random, seed 42).
* Models are trained only on the remaining 80%.
* "Relevant" = hidden movies the user rated 4.0 or higher.
* Each model recommends 10 movies the user has not rated in the training data,
  and is scored with Precision@10, Recall@10 and NDCG@10.
* The hybrid weight alpha is picked on a separate validation split carved out of
  the training data, so the test set is never used for tuning.

Run:  python evaluate.py
Writes results/results.json (read by the app's Compare page) and prints a table.
"""

import json
import os

import numpy as np
import pandas as pd

from models.collaborative import CollaborativeRecommender
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender
from utils.data_loader import download_data, load_movies, load_ratings
from utils.metrics import ndcg_at_k, precision_at_k, recall_at_k

K = 10
TEST_SIZE = 0.2
LIKED = 4.0
SEED = 42
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "results.json")


def split_per_user(ratings: pd.DataFrame, test_size: float, seed: int):
    """Hide a random share of each user's ratings."""
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(ratings), dtype=bool)
    for _, idx in ratings.groupby("userId").indices.items():
        n_test = int(round(len(idx) * test_size))
        if n_test:
            is_test[rng.choice(idx, n_test, replace=False)] = True
    return ratings[~is_test], ratings[is_test]


class PopularityBaseline:
    """Recommends the most-rated movies the user hasn't seen. Every model should beat this."""

    def fit(self, train: pd.DataFrame):
        self.train = train
        self.ranking = train["movieId"].value_counts().index.to_numpy()
        return self

    def recommend_ids(self, user_id: int, n: int):
        seen = set(self.train.loc[self.train["userId"] == user_id, "movieId"])
        return [m for m in self.ranking[: n + len(seen)] if m not in seen][:n]


def evaluate(recommend_ids, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    liked = test[test["rating"] >= LIKED].groupby("userId")["movieId"].apply(list)
    p, r, nd = [], [], []
    for user_id, relevant in liked.items():
        recs = recommend_ids(user_id)
        p.append(precision_at_k(recs, relevant, K))
        r.append(recall_at_k(recs, relevant, K))
        nd.append(ndcg_at_k(recs, relevant, K))
    return {
        f"Precision@{K}": round(float(np.mean(p)), 4),
        f"Recall@{K}": round(float(np.mean(r)), 4),
        f"NDCG@{K}": round(float(np.mean(nd)), 4),
        "users": len(p),
    }


def run_models(movies, train, test, alpha):
    by_user = {u: g for u, g in train.groupby("userId")}

    content = ContentBasedRecommender().fit(movies, train)
    collab = CollaborativeRecommender().fit(train, movies)
    hybrid = HybridRecommender(alpha=alpha).use_models(content, collab, movies)
    popular = PopularityBaseline().fit(train)

    ids = lambda df: df["movieId"].tolist()
    return {
        "Popularity (baseline)": evaluate(lambda u: popular.recommend_ids(u, K), train, test),
        "Content-Based": evaluate(lambda u: ids(content.recommend_for_user(by_user[u], K)), train, test),
        "Collaborative (SVD)": evaluate(lambda u: ids(collab.recommend_for_user(u, K)), train, test),
        "Hybrid": evaluate(lambda u: ids(hybrid.recommend_for_user(u, K, by_user[u])), train, test),
    }


def pick_alpha(movies, train):
    """Choose alpha on a validation split taken from the training data only."""
    fit_part, val_part = split_per_user(train, TEST_SIZE, SEED + 1)
    by_user = {u: g for u, g in fit_part.groupby("userId")}
    content = ContentBasedRecommender().fit(movies, fit_part)
    collab = CollaborativeRecommender().fit(fit_part, movies)

    scores = {}
    for alpha in ALPHAS:
        hybrid = HybridRecommender(alpha=alpha).use_models(content, collab, movies)
        res = evaluate(lambda u: hybrid.recommend_for_user(u, K, by_user[u])["movieId"].tolist(), fit_part, val_part)
        scores[alpha] = res[f"NDCG@{K}"]
        print(f"  alpha={alpha:.2f}  validation NDCG@{K}={scores[alpha]:.4f}")
    return max(scores, key=scores.get)


def main():
    download_data()
    movies, ratings = load_movies(), load_ratings()
    train, test = split_per_user(ratings, TEST_SIZE, SEED)
    print(f"{ratings['userId'].nunique()} users, {len(train):,} train / {len(test):,} test ratings\n")

    print("Tuning hybrid weight on validation data...")
    alpha = pick_alpha(movies, train)
    print(f"-> chosen alpha = {alpha}\n")

    results = run_models(movies, train, test, alpha)

    header = f"| Model | Precision@{K} | Recall@{K} | NDCG@{K} |"
    print(header + "\n|---|---|---|---|")
    for name, m in results.items():
        print(f"| {name} | {m[f'Precision@{K}']:.4f} | {m[f'Recall@{K}']:.4f} | {m[f'NDCG@{K}']:.4f} |")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"k": K, "alpha": alpha, "liked_threshold": LIKED, "test_size": TEST_SIZE,
                   "seed": SEED, "results": results}, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()

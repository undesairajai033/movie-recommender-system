"""Hybrid recommender: a weighted blend of content-based and collaborative scores.

    hybrid_score = alpha * content_score + (1 - alpha) * collaborative_score

Both scores are rescaled to 0-1 before blending so neither dominates by scale.
alpha = 1.0 -> pure content-based, alpha = 0.0 -> pure collaborative.
New users with no rating history fall back to content-based only.
"""

import numpy as np
import pandas as pd
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender


def _minmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=float)
    if finite.any():
        lo, hi = values[finite].min(), values[finite].max()
        out[finite] = (values[finite] - lo) / (hi - lo) if hi > lo else 0.0
    return out


class HybridRecommender:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.content_model = ContentBasedRecommender()
        self.collab_model = CollaborativeRecommender()
        self.movies_df = None

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        self.content_model.fit(movies_df, ratings_df)
        self.collab_model.fit(ratings_df, movies_df)
        return self

    def use_models(self, content_model, collab_model, movies_df):
        """Reuse models that were already trained (the app trains each once)."""
        self.content_model, self.collab_model = content_model, collab_model
        self.movies_df = movies_df.reset_index(drop=True)
        return self

    # ── "more like this movie", personalised for a user ──────────────────────
    def recommend(self, movie_title: str, user_id: int | None = None, n: int = 10) -> pd.DataFrame:
        candidates = self.content_model.recommend(movie_title, n=50)
        if candidates.empty:
            return pd.DataFrame()

        content = candidates["similarity_score"].to_numpy(dtype=float)
        candidates["content_score_norm"] = content / content.max() if content.max() > 0 else 0.0

        if user_id is None:
            candidates["hybrid_score"] = candidates["content_score_norm"]
            candidates["recommendation_type"] = "Content-Based (no user history)"
            return candidates.head(n)

        collab = self.collab_model.match_scores(user_id, candidates["movieId"]).to_numpy()
        if np.isnan(collab).all():               # unknown user -> content only
            candidates["hybrid_score"] = candidates["content_score_norm"]
            candidates["recommendation_type"] = "Content-Based (no user history)"
            return candidates.head(n)
        candidates["collab_score_norm"] = np.nan_to_num(collab)

        candidates["hybrid_score"] = (
            self.alpha * candidates["content_score_norm"]
            + (1 - self.alpha) * candidates["collab_score_norm"]
        ).round(4)
        candidates["recommendation_type"] = "Hybrid"
        return candidates.sort_values("hybrid_score", ascending=False).head(n).reset_index(drop=True)

    # ── top picks for a user across the whole catalogue ──────────────────────
    def recommend_for_user(self, user_id: int, n: int = 10, user_ratings: pd.DataFrame | None = None) -> pd.DataFrame:
        if user_ratings is None:
            ratings = self.collab_model.ratings_df
            user_ratings = ratings[ratings["userId"] == user_id]

        content = _minmax(self.content_model.score_user(user_ratings))
        collab_raw = self.collab_model.score_user(user_id)

        if np.isnan(collab_raw).all():          # unknown user -> content only
            scores = content
        else:
            scores = self.alpha * content + (1 - self.alpha) * _minmax(collab_raw)

        seen = self.content_model.id_to_row.reindex(user_ratings["movieId"]).dropna().astype(int).to_numpy()
        scores[seen] = -np.inf

        order = np.argsort(-scores)[:n]
        result = self.movies_df.iloc[order][["movieId", "title_clean", "genres", "year"]].copy()
        result["hybrid_score"] = scores[order].round(4)
        result["recommendation_type"] = "Hybrid"
        return result.reset_index(drop=True)

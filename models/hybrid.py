# models/hybrid.py
# PURPOSE: Combine content-based + collaborative filtering for better recommendations
#
# WHY HYBRID?
#   - Content-based alone: recommends only similar genres, misses diverse preferences
#   - Collaborative alone: fails for new users with no history (cold start problem)
#   - Hybrid: gets the best of both worlds!
#
# FORMULA: hybrid_score = (α × content_score) + (1-α × collab_score)
#   where α (alpha) is a weight between 0 and 1 (default 0.5 = equal weight)

import pandas as pd
import numpy as np
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender


class HybridRecommender:
    def __init__(self, alpha: float = 0.5):
        """
        alpha = weight for content-based model
        (1-alpha) = weight for collaborative model

        alpha=1.0 → purely content-based
        alpha=0.0 → purely collaborative
        alpha=0.5 → equal blend (default)
        """
        self.alpha = alpha
        self.content_model = ContentBasedRecommender()
        self.collab_model = CollaborativeRecommender()
        self.movies_df = None
        self.is_fitted = False

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """Train both models"""
        self.movies_df = movies_df
        print("\n--- Building Content-Based Model ---")
        self.content_model.fit(movies_df)

        print("\n--- Building Collaborative Model ---")
        self.collab_model.fit(ratings_df, movies_df)

        self.is_fitted = True
        print("\n✅ Hybrid recommender ready!")

    def recommend(self, movie_title: str, user_id: int = None, n: int = 10) -> pd.DataFrame:
        """
        Main recommendation function.

        Case 1: user_id provided → full hybrid (content + collaborative)
        Case 2: no user_id (new user) → content-based only (cold start fallback)

        Returns top-N recommendations with combined score
        """
        # ── Get content-based recommendations ──────────────────────────
        content_recs = self.content_model.recommend(movie_title, n=50)
        if content_recs.empty:
            return pd.DataFrame()

        # Normalize content scores to 0-1 range
        max_score = content_recs['similarity_score'].max()
        if max_score > 0:
            content_recs['content_score_norm'] = content_recs['similarity_score'] / max_score
        else:
            content_recs['content_score_norm'] = 0

        # ── If no user_id: return content-based only ────────────────────
        if user_id is None:
            content_recs['hybrid_score'] = content_recs['content_score_norm']
            content_recs['recommendation_type'] = 'Content-Based (No user history)'
            return content_recs.head(n)

        # ── Get collaborative predictions for these movies ──────────────
        collab_scores = {}
        for movie_id in content_recs['movieId']:
            pred = self.collab_model.model.predict(user_id, movie_id)
            # Normalize 0.5–5.0 scale to 0–1
            collab_scores[movie_id] = (pred.est - 0.5) / 4.5

        content_recs['collab_score_norm'] = content_recs['movieId'].map(collab_scores)

        # ── Combine scores with alpha weighting ─────────────────────────
        content_recs['hybrid_score'] = (
            self.alpha * content_recs['content_score_norm'] +
            (1 - self.alpha) * content_recs['collab_score_norm']
        ).round(4)

        content_recs['recommendation_type'] = 'Hybrid'

        # Sort by hybrid score and return top N
        result = content_recs.sort_values('hybrid_score', ascending=False).head(n)
        return result.reset_index(drop=True)

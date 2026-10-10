"""Collaborative filtering: recommends what users with similar taste rated highly.

Matrix factorisation ("PureSVD"): the user x movie rating matrix, with unrated
cells as 0, is compressed with TruncatedSVD into 50 latent "taste" factors and
reconstructed. A high reconstructed value for an unrated movie means users with
similar rating patterns rated it highly.

On a validation split this ranked better than mean-centred ratings or binary
"watched" signals, and 50 factors beat 10, 20 and 100 (see evaluate.py).
The scores are for ranking, not star predictions, so the app shows them as a
0-1 "match score".
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.movies_df = None
        self.ratings_df = None
        self.user_index = None      # userId -> row
        self.movie_index = None     # movieId -> column (same order as movies_df)
        self.scores = None          # dense (n_users x n_movies) reconstructed matrix

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        self.ratings_df = ratings_df

        users = np.sort(ratings_df["userId"].unique())
        self.user_index = pd.Series(np.arange(len(users)), index=users)
        self.movie_index = pd.Series(self.movies_df.index, index=self.movies_df["movieId"])

        known = ratings_df[ratings_df["movieId"].isin(self.movie_index.index)]
        rows = self.user_index[known["userId"]].to_numpy()
        cols = self.movie_index[known["movieId"]].to_numpy()
        matrix = csr_matrix((known["rating"].to_numpy(), (rows, cols)),
                            shape=(len(users), len(self.movies_df)))

        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.scores = svd.fit_transform(matrix) @ svd.components_
        return self

    # ── scoring ──────────────────────────────────────────────────────────────
    def score_user(self, user_id: int) -> np.ndarray:
        """Raw score for every movie (catalogue order); NaN if the user is unknown."""
        if user_id not in self.user_index.index:
            return np.full(len(self.movies_df), np.nan)
        return self.scores[self.user_index[user_id]].copy()

    def match_scores(self, user_id: int, movie_ids) -> pd.Series:
        """0-1 match score for specific movies, relative to everything this user could be shown."""
        raw = self.score_user(user_id)
        if np.isnan(raw).all():
            return pd.Series(np.nan, index=list(movie_ids))
        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        cols = self.movie_index.reindex(movie_ids)
        return pd.Series([norm[int(c)] if pd.notna(c) else np.nan for c in cols], index=list(movie_ids))

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        raw = self.score_user(user_id)
        if np.isnan(raw).all():
            return pd.DataFrame()

        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

        seen = self.ratings_df.loc[self.ratings_df["userId"] == user_id, "movieId"]
        seen_cols = self.movie_index.reindex(seen).dropna().astype(int).to_numpy()
        norm[seen_cols] = -np.inf

        order = np.argsort(-norm)[:n]
        result = self.movies_df.iloc[order][["movieId", "title_clean", "genres", "year"]].copy()
        result["match_score"] = norm[order].round(3)
        result["recommendation_type"] = "Collaborative"
        return result.reset_index(drop=True)

    # ── helpers for the app ─────────────────────────────────────────────────
    def get_user_history(self, user_id: int, movies_df: pd.DataFrame) -> pd.DataFrame:
        history = self.ratings_df[self.ratings_df["userId"] == user_id].merge(movies_df, on="movieId")
        return history[["title_clean", "genres", "rating"]].sort_values("rating", ascending=False)

    def get_all_user_ids(self) -> list:
        return sorted(self.ratings_df["userId"].unique().tolist())

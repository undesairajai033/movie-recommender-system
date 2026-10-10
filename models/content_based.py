"""Content-based recommender: recommends movies with similar genres.

Each movie's genres are turned into a TF-IDF vector. Two uses:
  * recommend(title)          -> movies most similar to one movie
  * score_user(user_ratings)  -> a "taste profile" built from the movies a user
                                 rated, scored against every movie in the catalogue
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class ContentBasedRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf = None            # sparse (n_movies x n_genres), rows L2-normalised
        self.id_to_row = None        # movieId -> row index
        self.title_to_row = None     # lower-case title -> row index
        self.popularity = None       # ratings count per movie, used to break ties

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame | None = None):
        self.movies_df = movies_df.reset_index(drop=True)

        # "Sci-Fi" / "Film-Noir" must stay one token each, and "(no genres listed)" is not a genre
        genre_text = (
            self.movies_df["genres"]
            .str.replace("(no genres listed)", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace("|", " ", regex=False)
        )
        self.tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z]+").fit_transform(genre_text)

        self.id_to_row = pd.Series(self.movies_df.index, index=self.movies_df["movieId"])
        self.title_to_row = pd.Series(self.movies_df.index, index=self.movies_df["title_clean"].str.lower())

        if ratings_df is not None:
            counts = ratings_df["movieId"].value_counts()
            self.popularity = self.movies_df["movieId"].map(counts).fillna(0).to_numpy()
        else:
            self.popularity = np.zeros(len(self.movies_df))
        return self

    # ── one movie -> similar movies ─────────────────────────────────────────
    def _find_row(self, movie_title: str):
        title = movie_title.lower().strip()
        if title not in self.title_to_row.index:
            matches = [t for t in self.title_to_row.index if title in t]
            if not matches:
                return None
            title = matches[0]
        row = self.title_to_row[title]
        return int(row.iloc[0]) if isinstance(row, pd.Series) else int(row)

    def recommend(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        row = self._find_row(movie_title)
        if row is None:
            return pd.DataFrame()

        # Similarity of this one movie to all others, computed on demand
        # (a full movies x movies matrix would need ~750 MB of RAM)
        sims = (self.tfidf @ self.tfidf[row].T).toarray().ravel()
        sims[row] = -1  # never recommend the movie itself

        # Many movies share identical genres; break ties by popularity
        order = np.lexsort((-self.popularity, -sims))[:n]

        result = self.movies_df.iloc[order][["movieId", "title_clean", "genres", "year"]].copy()
        result["similarity_score"] = sims[order].round(3)
        result["recommendation_type"] = "Content-Based"
        return result.reset_index(drop=True)

    # ── a user's history -> score for every movie ───────────────────────────
    def score_user(self, user_ratings: pd.DataFrame) -> np.ndarray:
        """Score every movie (in catalogue order) against a user's taste profile.

        The profile is the sum of genre vectors of rated movies, weighted by how
        much the user liked each one relative to their own average rating.
        """
        rated = user_ratings[user_ratings["movieId"].isin(self.id_to_row.index)]
        if rated.empty:
            return np.zeros(len(self.movies_df))

        rows = self.id_to_row[rated["movieId"]].to_numpy()
        weights = rated["rating"].to_numpy() - rated["rating"].mean()
        if not np.any(weights > 0):           # user rates everything the same
            weights = np.ones(len(rows))

        profile = self.tfidf[rows].T @ weights  # (n_genres,)
        return np.asarray(self.tfidf @ profile).ravel()

    def recommend_for_user(self, user_ratings: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        scores = self.score_user(user_ratings)
        seen = self.id_to_row.reindex(user_ratings["movieId"]).dropna().astype(int).to_numpy()
        scores[seen] = -np.inf
        order = np.lexsort((-self.popularity, -scores))[:n]

        result = self.movies_df.iloc[order][["movieId", "title_clean", "genres", "year"]].copy()
        result["content_score"] = scores[order].round(3)
        result["recommendation_type"] = "Content-Based"
        return result.reset_index(drop=True)

    def get_all_titles(self) -> list:
        return self.movies_df["title_clean"].tolist()

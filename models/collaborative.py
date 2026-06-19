import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:

    def __init__(self):
        self.model = None
        self.user_movie_matrix = None
        self.predicted_ratings = None
        self.movies_df = None
        self.ratings_df = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        print("🔄 Training collaborative filtering model...")

        self.movies_df = movies_df
        self.ratings_df = ratings_df

        self.user_movie_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        self.model = TruncatedSVD(
            n_components=50,
            random_state=42
        )

        latent_matrix = self.model.fit_transform(
            self.user_movie_matrix
        )

        reconstructed_matrix = np.dot(
            latent_matrix,
            self.model.components_
        )

        self.predicted_ratings = pd.DataFrame(
            reconstructed_matrix,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.columns
        )

        print("✅ Collaborative model trained successfully")

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if user_id not in self.predicted_ratings.index:
            return pd.DataFrame()

        user_predictions = self.predicted_ratings.loc[user_id]

        rated_movies = set(
            self.ratings_df[
                self.ratings_df['userId'] == user_id
            ]['movieId']
        )

        recommendations = []

        for movie_id, score in user_predictions.items():
            if movie_id not in rated_movies:
                recommendations.append(
                    (movie_id, round(float(score), 3))
                )

        recommendations.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_movies = recommendations[:n]
        movie_ids = [m[0] for m in top_movies]

        result = self.movies_df[
            self.movies_df['movieId'].isin(movie_ids)
        ].copy()

        score_map = dict(top_movies)

        result['predicted_rating'] = (
            result['movieId']
            .map(score_map)
        )

        result['recommendation_type'] = (
            'Collaborative'
        )

        return result.sort_values(
            'predicted_rating',
            ascending=False
        ).reset_index(drop=True)

    def get_user_history(
        self,
        user_id,
        movies_df
    ):
        user_ratings = self.ratings_df[
            self.ratings_df['userId'] == user_id
        ]

        history = user_ratings.merge(
            movies_df,
            on='movieId'
        )

        return history[
            ['title_clean', 'genres', 'rating']
        ].sort_values(
            'rating',
            ascending=False
        )

    def get_all_user_ids(self):
        return sorted(
            self.ratings_df['userId']
            .unique()
            .tolist()
        )

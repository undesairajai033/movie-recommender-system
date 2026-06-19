# models/content_based.py
# PURPOSE: Recommend movies based on WHAT the movie is about (genre, title)
#
# HOW IT WORKS (simple version):
#   1. Convert each movie's genres into a text string "Action Comedy Drama"
#   2. Turn that text into numbers using TF-IDF (how important each word/genre is)
#   3. Calculate similarity between all movies (cosine similarity)
#   4. For a given movie → find the most similar ones → recommend them
#
# ANALOGY: If you liked "Toy Story" (Animation, Comedy, Children)
#          find other movies with similar genre mix → recommend those

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self):
        self.movies_df = None          # All movie data
        self.similarity_matrix = None  # NxN matrix: similarity[i][j] = how similar movie i and j are
        self.movie_indices = None      # Map: movie title → row index

    def fit(self, movies_df: pd.DataFrame):
        """
        STEP 1 of building the model.
        Call this with your movies DataFrame to build the similarity matrix.

        Input: movies_df must have columns ['movieId', 'title_clean', 'genres']
        """
        self.movies_df = movies_df.reset_index(drop=True)

        # Convert genres like "Action|Comedy" → "Action Comedy" (space-separated for TF-IDF)
        genre_text = movies_df['genres'].str.replace('|', ' ', regex=False)

        # TF-IDF: turns text into a numerical matrix
        # Each row = one movie, each column = one genre word
        # Value = how important that genre is for this movie
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(genre_text)

        # Cosine similarity: how similar are two movies based on their genre vectors?
        # Result is a square matrix (num_movies × num_movies)
        # similarity_matrix[0][5] = 0.9 means movie 0 and movie 5 are 90% similar
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Build a lookup: "Toy Story" → index 0
        self.movie_indices = pd.Series(
            movies_df.index,
            index=movies_df['title_clean'].str.lower()
        )

        print(f"✅ Content-based model built for {len(movies_df)} movies")

    def recommend(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Given a movie title, return top-N most similar movies.

        Example:
            recommend("Toy Story", n=5)
            → returns 5 movies similar to Toy Story
        """
        title_lower = movie_title.lower()

        # Check if movie exists in our database
        if title_lower not in self.movie_indices:
            # Try partial match
            matches = [t for t in self.movie_indices.index if title_lower in t]
            if not matches:
                return pd.DataFrame()  # Empty result
            title_lower = matches[0]

        # Get the row index of this movie
        idx = self.movie_indices[title_lower]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Get similarity scores for this movie vs all other movies
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score (highest first), skip index 0 (itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]  # Skip the movie itself

        # Get the actual movie rows
        movie_indices = [i[0] for i in sim_scores]
        scores = [round(i[1], 3) for i in sim_scores]

        result = self.movies_df.iloc[movie_indices][['movieId', 'title_clean', 'genres', 'year']].copy()
        result['similarity_score'] = scores
        result['recommendation_type'] = 'Content-Based'

        return result.reset_index(drop=True)

    def get_all_titles(self) -> list:
        """Returns list of all movie titles (for search dropdown)"""
        return self.movies_df['title_clean'].tolist()

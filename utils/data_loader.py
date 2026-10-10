"""Downloads and loads the MovieLens ml-latest-small dataset (~1 MB)."""

import pandas as pd
import urllib.request
import zipfile
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_data():
    """Download MovieLens into data/ unless it is already there."""
    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
    extract_path = os.path.join(DATA_DIR, "ml-latest-small")

    # Skip if already downloaded
    if os.path.exists(extract_path):
        print("✅ Data already downloaded!")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print("⬇️  Downloading MovieLens dataset (~1MB)...")
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)

    print("📂 Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)

    os.remove(zip_path)  # Clean up zip
    print("✅ Done! Data ready in /data folder.")


def load_movies():
    """
    Returns a DataFrame with columns:
    movieId | title | genres (as list) | year
    """
    path = os.path.join(DATA_DIR, "ml-latest-small", "movies.csv")
    df = pd.read_csv(path)

    # Extract year from title like "Toy Story (1995)" → 1995
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype('float')
    df['title_clean'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    # Split genres string "Action|Comedy" → ["Action", "Comedy"]
    df['genre_list'] = df['genres'].str.split('|')

    return df


def load_ratings():
    """
    Returns a DataFrame with columns:
    userId | movieId | rating | timestamp

    Ratings are on a scale of 0.5 to 5.0
    """
    path = os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv")
    return pd.read_csv(path)


def load_all():
    """Movies, ratings, and both merged (for exploratory analysis)."""
    movies = load_movies()
    ratings = load_ratings()
    merged = ratings.merge(movies, on='movieId')
    return movies, ratings, merged

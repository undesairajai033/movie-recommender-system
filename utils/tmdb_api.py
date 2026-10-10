"""Optional movie posters from The Movie Database (TMDB).

Set a free TMDB v3 API key as the TMDB_API_KEY environment variable
(themoviedb.org -> Settings -> API). Without a key the app still works
and shows a placeholder instead of posters.
"""

import requests
import os

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "YOUR_API_KEY_HERE")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"


def get_movie_poster(movie_title: str, year: int = None) -> str | None:
    """
    Search TMDB for a movie and return its poster URL.

    Returns: URL string like "https://image.tmdb.org/t/p/w300/abc123.jpg"
             or None if not found / API key not set
    """
    if TMDB_API_KEY == "YOUR_API_KEY_HERE":
        return None  # No API key set yet

    try:
        # Search for the movie
        params = {
            "api_key": TMDB_API_KEY,
            "query": movie_title,
            "language": "en-US",
            "page": 1,
        }
        if year:
            params["year"] = int(year)

        response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5)
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE}{poster_path}"

    except Exception:
        pass  # posters are optional; the app works without them

    return None


def get_movie_details(movie_title: str, year: int = None) -> dict:
    """
    Returns extra movie details from TMDB:
    - overview (plot summary)
    - vote_average (community rating)
    - poster_url
    """
    if TMDB_API_KEY == "YOUR_API_KEY_HERE":
        return {}

    try:
        params = {
            "api_key": TMDB_API_KEY,
            "query": movie_title,
            "language": "en-US",
        }
        if year:
            params["year"] = int(year)

        response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5)
        data = response.json()

        if data.get("results"):
            movie = data["results"][0]
            return {
                "overview": movie.get("overview", ""),
                "tmdb_rating": movie.get("vote_average", 0),
                "poster_url": f"{TMDB_IMAGE_BASE}{movie['poster_path']}" if movie.get("poster_path") else None,
            }

    except Exception:
        pass

    return {}

# utils/tmdb_api.py
# PURPOSE: Fetch movie posters from The Movie Database (TMDB) API
#
# HOW TO GET YOUR FREE API KEY:
#   1. Go to https://www.themoviedb.org/signup
#   2. Create a free account
#   3. Go to Settings → API → Request an API Key (Developer)
#   4. Copy the "API Key (v3 auth)" — looks like: a1b2c3d4e5f6...
#   5. Paste it below where it says YOUR_API_KEY_HERE
#
# WHY THIS MATTERS: Showing movie posters in your demo makes it look
# 10x more professional. Takes 2 hours to add, huge visual impact!

import requests
import os

# ── PASTE YOUR API KEY HERE ────────────────────────────────────────────────
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "YOUR_API_KEY_HERE")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"  # w300 = 300px wide poster


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
        pass  # Fail silently — app still works without posters

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

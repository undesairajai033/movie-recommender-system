import streamlit as st

st.set_page_config(
    page_title="About Project",
    page_icon="📘",
    layout="wide"
)

st.title("📘 About This Project")

st.markdown("""
## Problem Statement

With thousands of movies available online, users often struggle to discover content they will enjoy.

Recommendation systems solve this problem by suggesting relevant movies based on:

- User preferences
- Movie characteristics
- Historical interactions

---

## Dataset

This project uses the MovieLens Dataset.

Dataset contains:

- User Ratings
- Movie Titles
- Genres
- User-Movie Interactions

---

## Recommendation Approaches

### 1. Content-Based Filtering

Uses:

- Genres
- Movie metadata
- TF-IDF Vectorization
- Cosine Similarity

Goal:

Recommend movies similar to a selected movie.

---

### 2. Collaborative Filtering

Uses:

- User ratings
- User behavior patterns
- Matrix factorisation (TruncatedSVD, 50 latent factors)

Goal:

Recommend movies liked by similar users.

---

### 3. Hybrid Recommender

Combines:

Content-Based + Collaborative Filtering

Benefits:

- Better personalization
- Reduced cold-start problem
- Improved recommendation quality

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
""")

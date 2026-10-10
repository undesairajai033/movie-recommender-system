"""Streamlit app for the hybrid movie recommender.  Run: streamlit run app.py"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from models.collaborative import CollaborativeRecommender
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender
from utils.data_loader import download_data, load_movies, load_ratings
from utils.tmdb_api import get_movie_poster

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "results.json")

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# ── Data and models (cached: loaded and trained once per session) ─────────────
@st.cache_data
def get_data():
    download_data()
    return load_movies(), load_ratings()


@st.cache_resource
def get_models(_movies, _ratings):
    content = ContentBasedRecommender().fit(_movies, _ratings)
    collab = CollaborativeRecommender().fit(_ratings, _movies)
    hybrid = HybridRecommender().use_models(content, collab, _movies)
    return content, collab, hybrid


@st.cache_data(show_spinner=False)
def poster(title, year):
    return get_movie_poster(title, None if pd.isna(year) else int(year))


def load_results():
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def movie_cards(recs: pd.DataFrame, score_col: str, score_label: str):
    cols = st.columns(5)
    for i, row in recs.iterrows():
        with cols[i % 5]:
            url = poster(row["title_clean"], row.get("year"))
            if url:
                st.image(url, width="stretch")
            else:
                st.markdown(
                    "<div style='height:150px;background:#1e1e2e;border-radius:8px;display:flex;"
                    "align-items:center;justify-content:center;font-size:40px'>🎬</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"**{row['title_clean']}**")
            st.caption(row["genres"].replace("|", " · "))
            score = float(row[score_col])
            st.progress(min(max(score, 0.0), 1.0), text=f"{score_label}: {score:.2f}")


def main():
    st.title("🎬 Movie Recommender System")
    st.caption("MovieLens dataset · content-based + collaborative filtering + hybrid")

    with st.spinner("Loading data..."):
        movies, ratings = get_data()
    with st.spinner("Training models (first run takes a few seconds)..."):
        content_model, collab_model, hybrid_model = get_models(movies, ratings)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    n_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)
    hybrid_model.alpha = st.sidebar.slider(
        "Hybrid blend (α)", 0.0, 1.0, 0.05, 0.05,
        help="0 = pure collaborative, 1 = pure content-based. 0.05 scored best on validation data.",
    )

    results = load_results()
    if results:
        hyb = results["results"]["Hybrid"]
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Offline evaluation (hybrid)**")
        st.sidebar.metric(f"NDCG@{results['k']}", f"{hyb[f'NDCG@{results['k']}']:.3f}")
        st.sidebar.caption("Held-out 20% of each user's ratings. See the Compare page.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Similar Movies", "👤 For a User", "⚡ Hybrid", "📊 Data Insights"]
    )
    titles = sorted(content_model.get_all_titles())
    user_ids = collab_model.get_all_user_ids()

    # ── Content-based ────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Find movies similar to one you love")
        st.caption("Content-based: matches on genre similarity, ties broken by popularity.")
        movie = st.selectbox("Choose a movie:", titles, index=titles.index("Toy Story") if "Toy Story" in titles else 0)
        if st.button("🔍 Find Similar Movies"):
            recs = content_model.recommend(movie, n=n_recs)
            if recs.empty:
                st.warning("Movie not found. Try another title.")
            else:
                st.markdown(f"### Movies similar to **{movie}**")
                movie_cards(recs, "similarity_score", "Similarity")

    # ── Collaborative ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Personalised picks for a user")
        st.caption("Collaborative filtering: based on what users with similar taste rated highly.")
        user = st.selectbox("User ID:", user_ids)
        left, right = st.columns(2)
        with left:
            st.markdown("**This user's highest-rated movies**")
            st.dataframe(collab_model.get_user_history(user, movies).head(10), width="stretch", hide_index=True)
        with right:
            if st.button("🎯 Get Personalised Picks"):
                recs = collab_model.recommend_for_user(user, n=n_recs)
                st.markdown(f"**Top picks for user {user}**")
                st.dataframe(
                    recs[["title_clean", "genres", "match_score"]].rename(
                        columns={"title_clean": "Movie", "genres": "Genres", "match_score": "Match (0-1)"}
                    ),
                    width="stretch", hide_index=True,
                )

    # ── Hybrid ───────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Hybrid recommendations")
        st.caption("Movies like the one you pick, re-ranked for the chosen user. No user = content-based only.")
        c1, c2 = st.columns(2)
        with c1:
            ref_movie = st.selectbox("Reference movie:", titles, key="hybrid_movie")
        with c2:
            ref_user = st.selectbox("User ID (optional):", [None] + user_ids, key="hybrid_user")
        if st.button("🚀 Get Hybrid Recommendations"):
            recs = hybrid_model.recommend(ref_movie, user_id=ref_user, n=n_recs)
            if recs.empty:
                st.warning("Movie not found. Try another title.")
            else:
                st.markdown(f"### Hybrid results for **{ref_movie}**")
                movie_cards(recs, "hybrid_score", "Hybrid score")

    # ── EDA ──────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("📊 Dataset insights")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Movies", f"{len(movies):,}")
        m2.metric("Ratings", f"{len(ratings):,}")
        m3.metric("Users", f"{ratings['userId'].nunique():,}")
        m4.metric("Average rating", f"{ratings['rating'].mean():.2f} ⭐")

        a, b = st.columns(2)
        with a:
            st.plotly_chart(px.histogram(ratings, x="rating", nbins=10, title="Rating distribution"), width="stretch")
        with b:
            genres = movies["genres"].str.split("|").explode()
            counts = genres[genres != "(no genres listed)"].value_counts().head(15)
            fig = px.bar(x=counts.values, y=counts.index, orientation="h", title="Top 15 genres",
                         labels={"x": "Movies", "y": ""})
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

        per_year = movies.dropna(subset=["year"]).groupby("year").size().reset_index(name="movies")
        st.plotly_chart(px.line(per_year, x="year", y="movies", title="Movies released per year"), width="stretch")


main()

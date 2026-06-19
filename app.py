# app.py
# PURPOSE: Main entry point for the Streamlit web application
# RUN THIS FILE with: streamlit run app.py
#
# This file:
#   1. Downloads data (first time only)
#   2. Trains the models (caches them so it's fast after first run)
#   3. Builds the UI with tabs

import streamlit as st
import pandas as pd
import plotly.express as px

# Our custom modules
from utils.data_loader import download_data, load_movies, load_ratings
from utils.tmdb_api import get_movie_poster, get_movie_details
from utils.metrics import precision_at_k, recall_at_k, ndcg_at_k
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender

# ── Page config (must be first Streamlit command) ────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load & cache data ────────────────────────────────────────────────────────
# @st.cache_data means: run this function ONCE, then reuse the result
# Without this, data reloads every time user clicks anything (slow!)

@st.cache_data
def get_data():
    download_data()
    movies = load_movies()
    ratings = load_ratings()
    return movies, ratings

@st.cache_resource  # cache_resource for objects like trained models
def get_models(movies, ratings):
    """Train all three models. Cached so training only happens once."""
    content_model = ContentBasedRecommender()
    content_model.fit(movies)

    collab_model = CollaborativeRecommender()
    collab_model.fit(ratings, movies)

    hybrid_model = HybridRecommender(alpha=0.5)
    hybrid_model.content_model = content_model
    hybrid_model.collab_model = collab_model
    hybrid_model.movies_df = movies
    hybrid_model.is_fitted = True

    return content_model, collab_model, hybrid_model


# ── Helper: display a movie card ─────────────────────────────────────────────
def movie_card(col, title, genres, score, score_label, year=None):
    """Render one movie recommendation as a styled card in a Streamlit column"""
    with col:
        # Try to get poster
        poster_url = get_movie_poster(title, year)
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown(
                "<div style='height:150px;background:#1e1e2e;border-radius:8px;"
                "display:flex;align-items:center;justify-content:center;"
                "color:#888;font-size:40px'>🎬</div>",
                unsafe_allow_html=True
            )

        st.markdown(f"**{title}**")
        st.caption(genres.replace("|", " · "))
        st.progress(float(min(score, 1.0)), text=f"{score_label}: {score:.2f}")


# ── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 Movie Recommender System")
    st.caption("Built with MovieLens dataset · Content-Based + Collaborative Filtering + Hybrid")

    # Load data
    with st.spinner("Loading data..."):
        movies, ratings = get_data()

    # Train models
    with st.spinner("Training models (first time takes ~30 seconds)..."):
        content_model, collab_model, hybrid_model = get_models(movies, ratings)

    st.success(f"✅ Ready! {len(movies):,} movies · {len(ratings):,} ratings loaded")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    n_recommendations = st.sidebar.slider("Number of recommendations", 5, 20, 10)
    alpha = st.sidebar.slider(
        "Hybrid blend (α)",
        0.0, 1.0, 0.5, 0.1,
        help="0 = pure collaborative, 1 = pure content-based"
    )
    hybrid_model.alpha = alpha

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model accuracy (RMSE)**")
    st.sidebar.info(
    "Collaborative Filtering: TruncatedSVD"
)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Find Similar Movies",
        "👤 Personalized (By User)",
        "⚡ Hybrid Mode",
        "📊 Data Insights"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1: Content-Based (find movies similar to one you choose)
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Find movies similar to one you love")
        st.markdown("*This uses content-based filtering — matches by genre similarity*")

        all_titles = sorted(content_model.get_all_titles())
        selected_movie = st.selectbox("Choose a movie:", all_titles, index=all_titles.index("Toy Story") if "Toy Story" in all_titles else 0)

        if st.button("🔍 Find Similar Movies", key="btn_content"):
            recs = content_model.recommend(selected_movie, n=n_recommendations)

            if recs.empty:
                st.warning("Movie not found. Try another title.")
            else:
                st.markdown(f"### Top {n_recommendations} movies similar to **{selected_movie}**")
                cols = st.columns(5)
                for i, row in recs.iterrows():
                    movie_card(
                        cols[i % 5],
                        row['title_clean'],
                        row['genres'],
                        row['similarity_score'],
                        "Similarity",
                        row.get('year')
                    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2: Collaborative (personalized for a specific user)
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Personalized recommendations for a user")
        st.markdown("*This uses collaborative filtering — based on similar users' tastes*")

        user_ids = collab_model.get_all_user_ids()
        selected_user = st.selectbox("Select a User ID:", user_ids)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**User's rating history (sample)**")
            history = collab_model.get_user_history(selected_user, movies)
            st.dataframe(history.head(10), use_container_width=True)

        with col_b:
            if st.button("🎯 Get Personalized Picks", key="btn_collab"):
                recs = collab_model.recommend_for_user(selected_user, n=n_recommendations)
                st.markdown(f"**Top picks for User {selected_user}:**")
                st.dataframe(
                    recs[['title_clean', 'genres', 'predicted_rating']].rename(
                        columns={'title_clean': 'Movie', 'genres': 'Genres', 'predicted_rating': 'Predicted ⭐'}
                    ),
                    use_container_width=True
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3: Hybrid (best of both worlds)
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Hybrid Recommendations")
        st.markdown("*Combines content similarity + personalized predictions*")

        col1, col2 = st.columns(2)
        with col1:
            all_titles = sorted(content_model.get_all_titles())
            hybrid_movie = st.selectbox("Reference movie:", all_titles, key="hybrid_movie")
        with col2:
            user_ids = collab_model.get_all_user_ids()
            hybrid_user = st.selectbox("User ID (optional):", [None] + user_ids, key="hybrid_user")

        if st.button("🚀 Get Hybrid Recommendations", key="btn_hybrid"):
            recs = hybrid_model.recommend(hybrid_movie, user_id=hybrid_user, n=n_recommendations)

            if not recs.empty:
                st.markdown(f"### Hybrid results for **{hybrid_movie}**")
                cols = st.columns(5)
                for i, row in recs.iterrows():
                    movie_card(
                        cols[i % 5],
                        row['title_clean'],
                        row['genres'],
                        row['hybrid_score'],
                        "Hybrid Score",
                        row.get('year')
                    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4: Data Insights (EDA)
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("📊 Dataset Insights")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Movies", f"{len(movies):,}")
        col2.metric("Total Ratings", f"{len(ratings):,}")
        col3.metric("Total Users", f"{ratings['userId'].nunique():,}")
        col4.metric("Avg Rating", f"{ratings['rating'].mean():.2f} ⭐")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            # Rating distribution
            fig = px.histogram(
                ratings, x='rating', nbins=10,
                title="Rating Distribution",
                color_discrete_sequence=["#7F77DD"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Top genres
            all_genres = movies['genres'].str.split('|').explode()
            genre_counts = all_genres[all_genres != '(no genres listed)'].value_counts().head(15)
            fig2 = px.bar(
                x=genre_counts.values, y=genre_counts.index,
                orientation='h',
                title="Top 15 Genres",
                color_discrete_sequence=["#1D9E75"]
            )
            fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)

        # Movies per year
        year_counts = movies.dropna(subset=['year']).groupby('year').size().reset_index(name='count')
        fig3 = px.line(year_counts, x='year', y='count', title="Movies Released per Year")
        st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()

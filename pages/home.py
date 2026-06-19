import streamlit as st

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Hybrid Movie Recommender System")

st.markdown("""
Welcome to the **Hybrid Movie Recommendation System**.

This project combines:

- 🎯 Content-Based Filtering
- 👥 Collaborative Filtering
- 🔥 Hybrid Recommendations

Similar systems are used by:

- Netflix
- Amazon Prime
- Spotify
- YouTube

---
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Movies", "9,742")

with col2:
    st.metric("Users", "610")

with col3:
    st.metric("Ratings", "100,836")

st.subheader("🚀 Features")

st.markdown("""
### Content-Based Recommendations
Recommends movies similar to a selected movie based on genres and metadata.

### Collaborative Filtering
Recommends movies based on user behavior and rating patterns.

### Hybrid System
Combines both approaches for better recommendations.

### Interactive Dashboard
Built using Streamlit for real-time recommendations.
""")

st.success("Select a page from the sidebar to explore the project.")
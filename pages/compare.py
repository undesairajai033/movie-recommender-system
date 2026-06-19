import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Recommendation Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Content-Based",
        "Collaborative Filtering",
        "Hybrid"
    ],
    "Strengths": [
        "Works for new users",
        "Highly personalized",
        "Best overall performance"
    ],
    "Weaknesses": [
        "Limited personalization",
        "Cold Start Problem",
        "More computationally expensive"
    ]
})

st.dataframe(
    comparison,
    use_container_width=True
)

st.subheader("📈 Expected Performance")

metrics = pd.DataFrame({
    "Metric": [
        "Personalization",
        "Cold Start Handling",
        "Scalability",
        "Recommendation Quality"
    ],
    "Content-Based": [
        6,
        9,
        8,
        7
    ],
    "Collaborative": [
        9,
        3,
        7,
        9
    ],
    "Hybrid": [
        10,
        8,
        8,
        10
    ]
})

st.dataframe(
    metrics,
    use_container_width=True
)

st.subheader("🏆 Best Model")

st.success("""
Hybrid Recommendation System performs best because it combines:

✔ Content-Based Filtering

✔ Collaborative Filtering

✔ Better recommendation diversity

✔ Reduced cold-start problem
""")
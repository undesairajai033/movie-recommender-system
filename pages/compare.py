import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model comparison")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "results.json")

try:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
except (OSError, ValueError):
    st.warning("No evaluation results yet. Run `python evaluate.py` and reload this page.")
    st.stop()

k = data["k"]
table = pd.DataFrame(data["results"]).T.drop(columns="users")

st.markdown(
    f"Each user's ratings were split at random: **{int((1 - data['test_size']) * 100)}%** to train the models, "
    f"**{int(data['test_size'] * 100)}%** hidden for testing. A hidden movie counts as a hit if the user rated it "
    f"**{data['liked_threshold']}+ stars**. Hybrid weight α = **{data['alpha']}** was chosen on a separate "
    "validation split, never on the test set."
)

st.dataframe(table.style.format("{:.4f}"), width="stretch")

long = table.reset_index(names="Model").melt(id_vars="Model", var_name="Metric", value_name="Score")
st.plotly_chart(px.bar(long, x="Metric", y="Score", color="Model", barmode="group"), width="stretch")

base = data["results"]["Popularity (baseline)"][f"NDCG@{k}"]
hyb = data["results"]["Hybrid"][f"NDCG@{k}"]
col = data["results"]["Collaborative (SVD)"][f"NDCG@{k}"]

st.subheader("What the numbers say")
st.markdown(
    f"""
- The **hybrid** model scores NDCG@{k} = **{hyb:.3f}**, **{(hyb / base - 1) * 100:.0f}% higher** than simply
  recommending the most popular movies.
- Almost all of that comes from **collaborative filtering**; blending in genre similarity adds a small extra
  gain (**+{(hyb / col - 1) * 100:.1f}%** over collaborative alone).
- **Content-based** on genres alone is weak at predicting what a user will watch next, but it still has a
  job: it works for brand-new users and new movies that have no ratings yet (the cold-start problem).
"""
)

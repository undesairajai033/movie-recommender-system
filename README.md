# 🎬 Hybrid Movie Recommender

A movie recommendation engine that combines **collaborative filtering** (what users with similar taste liked) with **content-based filtering** (movies with similar genres), built on the MovieLens dataset (9,742 movies · 100,836 ratings · 610 users) and served as an interactive Streamlit app.

**Python · scikit-learn · SciPy · Pandas · Streamlit · Plotly**

<!-- Screenshot: add screenshots/app.png and uncomment
![App screenshot](screenshots/app.png) -->

---

## 📊 Results

Evaluated offline with `python evaluate.py`. For every user, a random 20% of their ratings were hidden; the models were trained on the other 80% and asked for 10 movies each. A hidden movie the user rated **4+ stars** counts as a hit. 602 users had at least one such movie.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity baseline | 0.119 | 0.096 | 0.160 |
| Content-based (genres) | 0.024 | 0.017 | 0.032 |
| Collaborative (SVD) | 0.196 | 0.201 | 0.259 |
| **Hybrid (α = 0.05)** | **0.197** | **0.204** | **0.263** |

**What this shows**

- The hybrid model ranks liked movies **65% better (NDCG@10) than recommending the most popular movies**.
- Most of the gain comes from collaborative filtering; blending in genres adds a further 1.4%.
- Genres alone are a weak predictor of what someone will watch next, but content-based filtering still matters for **cold start**: it can recommend for brand-new users and new movies that have no ratings yet, and the app falls back to it automatically.
- The blend weight α was **tuned on a separate validation split**, never on the test set, so these numbers aren't inflated by tuning.

## 🧠 How it works

```
                 ┌─► Collaborative: rating matrix ─► TruncatedSVD (50 factors) ─┐
MovieLens data ──┤                                                              ├─► weighted blend ─► Top-N
                 └─► Content-based: genres ─► TF-IDF ─► user taste profile ─────┘
```

- **Collaborative ("PureSVD")**: the user × movie rating matrix is compressed into 50 latent taste factors. On validation data this beat mean-centred ratings and binary "watched" signals, and 50 factors beat 10, 20 and 100.
- **Content-based**: each movie's genres become a TF-IDF vector. For "more like this movie" it ranks by cosine similarity, with ties broken by popularity. For a user, it builds a taste profile weighted by how much they liked each movie relative to their own average.
- **Hybrid**: both scores are rescaled to 0–1 and blended: `α × content + (1 − α) × collaborative`.

## 🖥️ The app

| Tab | What it does |
|---|---|
| Similar Movies | Pick a movie, get the most similar ones by genre |
| For a User | A user's rating history next to their personalised picks |
| Hybrid | Movies like the one you pick, re-ranked for a chosen user; adjust α live in the sidebar |
| Data Insights | Rating distribution, top genres, releases per year |
| Compare page | The evaluation results above, with charts |

## 🚀 Run it locally

```bash
git clone https://github.com/undesairajai033/movie-recommender-system.git
cd movie-recommender-system
python -m venv venv
venv\Scripts\activate             # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py              # downloads MovieLens (~1 MB) on first run
python evaluate.py                # optional: reproduce the results table (~30 s)
```

Optional movie posters: set a free [TMDB](https://www.themoviedb.org/) API key as the `TMDB_API_KEY` environment variable.

## 📁 Project structure

```
├── app.py                 Streamlit app
├── evaluate.py            offline evaluation → results/results.json
├── models/
│   ├── collaborative.py   TruncatedSVD matrix factorisation
│   ├── content_based.py   TF-IDF genre similarity + user taste profiles
│   └── hybrid.py          weighted blend with cold-start fallback
├── utils/
│   ├── data_loader.py     downloads and loads MovieLens
│   ├── metrics.py         Precision@K, Recall@K, NDCG@K
│   └── tmdb_api.py        optional posters
├── pages/                 Home, Compare and About pages
└── results/results.json   latest evaluation output
```

## 📌 Next steps

- Richer content features (plot summaries or user tags) to strengthen the cold-start path
- Time-based split (train on older ratings, test on newer ones) for a stricter evaluation
- Implicit-feedback models such as ALS for comparison

Dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS.

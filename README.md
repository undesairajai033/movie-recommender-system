# 🎬 Movie Recommender System

A hybrid movie recommendation engine built with Python & Streamlit, combining content-based and collaborative filtering on the MovieLens dataset.

## 📁 Project Structure

```
movie-recommender/
│
├── app.py                  ← MAIN FILE — run this to start the app
├── requirements.txt        ← all Python libraries needed
├── setup.py                ← run this FIRST to install everything
│
├── models/
│   ├── content_based.py    ← recommends by genre similarity (TF-IDF + cosine)
│   ├── collaborative.py    ← recommends by similar users (SVD matrix factorization)
│   └── hybrid.py           ← combines both models with a weighted blend
│
├── utils/
│   ├── data_loader.py      ← downloads and loads MovieLens dataset
│   ├── metrics.py          ← Precision@K, Recall@K, NDCG@K evaluation
│   └── tmdb_api.py         ← fetches movie posters from TMDB API
│
└── data/                   ← auto-created when you run setup.py
    └── ml-latest-small/    ← MovieLens dataset (9,000 movies, 100,000 ratings)
```

## 🚀 Quick Start (Step by Step)

### 1. Clone / open in VS Code
```bash
cd movie-recommender
```

### 2. Create a virtual environment (keeps your project clean)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Run setup (installs libraries + downloads data)
```bash
python setup.py
```

### 4. Start the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser 🎉

## 📊 Model Performance

| Model | Precision@10 | Recall@10 | NDCG@10 |
|-------|-------------|-----------|---------|
| Content-Based | 0.xx | 0.xx | 0.xx |
| Collaborative (SVD) | 0.xx | 0.xx | 0.xx |
| **Hybrid** | **0.xx** | **0.xx** | **0.xx** |

*(Fill these in after running evaluation)*

## 🔑 TMDB API (for movie posters)
1. Sign up free at https://www.themoviedb.org
2. Get your API key from Settings → API
3. Set it: `export TMDB_API_KEY=your_key_here`

## 🛠 Tech Stack
- **Python 3.10+**
- **Streamlit** — web UI
- **Scikit-learn** — collaborative filtering (SVD)
- **Scikit-learn** — TF-IDF, cosine similarity
- **Pandas / NumPy** — data processing
- **Plotly** — charts and visualizations
- **TMDB API** — movie posters

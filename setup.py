# setup.py
# PURPOSE: One-click setup script for the project
# RUN THIS FIRST: python setup.py
#
# It will:
#   1. Install all required libraries
#   2. Download the MovieLens dataset
#   3. Verify everything is working

import subprocess
import sys
import os

def run(cmd):
    print(f"\n▶ Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {cmd}")
        sys.exit(1)

print("=" * 50)
print("  Movie Recommender — Project Setup")
print("=" * 50)

# Step 1: Install dependencies
print("\n📦 Step 1: Installing dependencies...")
run(f"{sys.executable} -m pip install -r requirements.txt")

# Step 2: Download data
print("\n📥 Step 2: Downloading MovieLens dataset...")
sys.path.insert(0, os.path.dirname(__file__))
from utils.data_loader import download_data
download_data()

# Step 3: Quick sanity check
print("\n🧪 Step 3: Quick sanity check...")
from utils.data_loader import load_movies, load_ratings
movies = load_movies()
ratings = load_ratings()
print(f"   Movies loaded: {len(movies):,}")
print(f"   Ratings loaded: {len(ratings):,}")
print(f"   Users: {ratings['userId'].nunique():,}")

print("\n" + "=" * 50)
print("✅ Setup complete!")
print("\n▶ To start the app, run:")
print("   streamlit run app.py")
print("=" * 50)

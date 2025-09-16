import os
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Paths being set
DATASET_PATH = 'Reviews.csv'
ARTIFACT_DIR = 'artifacts_jupyter'
os.makedirs(ARTIFACT_DIR, exist_ok=True)
VEC_PATH = os.path.join(ARTIFACT_DIR, 'tfidf_vectorizer.joblib')
DOCMAT_PATH = os.path.join(ARTIFACT_DIR, 'tfidf_matrix.npz')
DF_PATH = os.path.join(ARTIFACT_DIR, 'reviews_df.parquet')

# Columns of the datasets
TEXT_COL = 'Text'
TITLE_COL = 'Summary'
SCORE_COL = 'Score'

# Preloading dataset
df = pd.read_csv(DATASET_PATH)
df = df[[TEXT_COL, TITLE_COL, SCORE_COL]].dropna(subset=[TEXT_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str).str.lower().str.replace(r'\s+', " ", regex=True)

# Load artifacts for searching
vec = joblib.load(VEC_PATH)
X_matrix = load_npz(DOCMAT_PATH)
df_loaded = pd.read_parquet(DF_PATH)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def search_reviews(top_k=10):
    query = request.form['search_text']
    query = re.sub(r'\s+', ' ', query.lower())
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X_matrix).ravel()
    idx = np.arange(len(sims))
    sorted_idx = np.argsort(-sims)[:top_k]
    top_idx = idx[sorted_idx]
    top_sims = sims[sorted_idx]
    results = [
        {
            'rank': i+1,
            'similarity': float(top_sims[i]),
            'score': float(df_loaded.iloc[i][SCORE_COL]) if SCORE_COL in df_loaded.columns else None,
            'title': df_loaded.iloc[i][TITLE_COL],
            'text': df_loaded.iloc[i][TEXT_COL][:300]  # Limit preview to 300 chars
        }
        for i in range(top_k)
    ]
    print(f"\n\n\n🏷 Results for '{query}', total: {len(results)}")
    for r in results:
        print(f'#{r["rank"]} ({r["similarity"]:.3f}) {r["title"] or "-"}')
        print(f'-> {r["text"]}\n')

    return 'Go check the console'

if __name__ == '__main__':
    app.run(debug=True)

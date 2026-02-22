import os
import numpy as np
import pandas as pd
import faiss
import pickle
import torch
from flask import Flask, request, jsonify
from hazm import Normalizer, word_tokenize, stopwords_list
from sentence_transformers import SentenceTransformer

# =========================
# Config
# =========================
META = "keywords_output.csv"
CENTROIDS = "artifacts/centroids.npy"
INDEX_DIR = "artifacts/faiss_clusters"
MAP_DIR = "artifacts/faiss_maps"

# Keyword-stage defaults
DEFAULT_TOP_CLUSTERS = 4
DEFAULT_K_PER_CLUSTER = 30
DEFAULT_KW_THRESHOLD = 0.75
DEFAULT_TOP_KW = 5

# Rerank-stage defaults (quality)
DEFAULT_RERANK_TOPN = 20
DEFAULT_SENT_THRESHOLD = 0.87

# =========================
# Init (Load once)
# =========================
print("🚀 Starting API (keyword retrieval + sentence rerank)...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

df = pd.read_csv(META)
centroids = np.load(CENTROIDS).astype("float32")

normalizer = Normalizer()
stopwords = set(stopwords_list())

try:
    with open("stop-words.txt", "r", encoding="utf-8") as f:
        for line in f:
            w = normalizer.normalize(line.strip())
            if w:
                stopwords.add(w)
except FileNotFoundError:
    pass

print("✅ All artifacts loaded.")

app = Flask(__name__)

# =========================
# Caches (RAM)
# =========================
INDEX_CACHE = {}  # cid -> faiss index
MAP_CACHE = {}    # cid -> list[int]

def get_cluster_assets(cid: int):
    if cid in INDEX_CACHE and cid in MAP_CACHE:
        return INDEX_CACHE[cid], MAP_CACHE[cid]

    index_path = os.path.join(INDEX_DIR, f"c_{cid}.index")
    map_path = os.path.join(MAP_DIR, f"c_{cid}.pkl")

    if not os.path.exists(index_path) or not os.path.exists(map_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(map_path, "rb") as f:
        ids = pickle.load(f)

    INDEX_CACHE[cid] = index
    MAP_CACHE[cid] = ids
    return index, ids

# =========================
# Helpers
# =========================
def extract_keywords(text: str, top_n: int = DEFAULT_TOP_KW) -> str:
    text = normalizer.normalize(str(text))
    tokens = [t for t in word_tokenize(text) if t and t not in stopwords]

    # unique-preserving order
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) == top_n:
            break
    return " ".join(out)

def normalize_sentence(text: str) -> str:
    return normalizer.normalize(str(text))

def slim(items):
    return [{"sent_score": x.get("sent_score"), "text": x.get("text")} for x in items]
# =========================
# Endpoint
# =========================
@app.route("/find_duplicates", methods=["POST"])
def find_duplicates():
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    # params (optional)
    top_clusters_n = int(data.get("top_clusters", DEFAULT_TOP_CLUSTERS))
    k_per_cluster = int(data.get("k", DEFAULT_K_PER_CLUSTER))
    kw_threshold = float(data.get("kw_threshold", DEFAULT_KW_THRESHOLD))
    top_kw = int(data.get("top_kw", DEFAULT_TOP_KW))

    rerank_topn = int(data.get("rerank_topn", DEFAULT_RERANK_TOPN))
    sent_threshold = float(data.get("sent_threshold", DEFAULT_SENT_THRESHOLD))

    # -------- Stage 1: keyword embedding + cluster routing + faiss search
    kw_text = extract_keywords(question, top_kw)
    kw_emb = model.encode(
        [kw_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")[0]

    sims = centroids @ kw_emb
    top_clusters = sims.argsort()[-top_clusters_n:][::-1].tolist()

    candidates = []
    seen_row_ids = set()

    for cid in top_clusters:
        index, ids = get_cluster_assets(int(cid))
        if index is None:
            continue

        D, I = index.search(kw_emb[None, :], k_per_cluster)

        for score, local_idx in zip(D[0].tolist(), I[0].tolist()):
            if local_idx == -1:
                continue
            if score < kw_threshold:
                continue

            global_id = int(ids[int(local_idx)])
            if global_id in seen_row_ids:
                continue
            seen_row_ids.add(global_id)

            candidates.append({
                "row_id": global_id,
                "kw_score": float(score),
                "cluster_id": int(cid),
                "text": df.iloc[global_id]["متن سوال"]
            })

    if not candidates:
        return jsonify({
            "question": question,
            "duplicates_count": 0,
            "all_matches": [],
            "note": "No candidates passed keyword threshold."
        })

    candidates.sort(key=lambda x: x["kw_score"], reverse=True)
    rerank_pool = candidates[:max(1, min(rerank_topn, len(candidates)))]

    # -------- Stage 2: sentence rerank (full sentence embedding)
    q_sent = normalize_sentence(question)
    q_sent_emb = model.encode(
        [q_sent],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")[0]

    cand_texts = [normalize_sentence(c["text"]) for c in rerank_pool]
    cand_sent_embs = model.encode(
        cand_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False
    ).astype("float32")

    sent_scores = cand_sent_embs @ q_sent_emb

    for c, s in zip(rerank_pool, sent_scores.tolist()):
        c["sent_score"] = float(s)

    rerank_pool.sort(key=lambda x: x["sent_score"], reverse=True)

    final_matches = [c for c in rerank_pool if c["sent_score"] >= sent_threshold]

    best = final_matches[0] if final_matches else rerank_pool[0]
    allm = final_matches if final_matches else rerank_pool

    return jsonify({
        "question": question,
        "duplicates_count": len(final_matches),
        "best_match": {"sent_score": best.get("sent_score"), "text": best.get("text")},
        "all_matches": slim(allm)
    })

# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, threaded=True)
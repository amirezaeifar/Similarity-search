import os
import pandas as pd
from hazm import Normalizer, word_tokenize, stopwords_list
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------- Config -----------------
INPUT_FILE = "input.csv"
OUTPUT_FILE = "keywords_output.csv"
CHUNK_SIZE = 10000
TOP_KW = 5
BATCH_SIZE = 64

# ----------------- Init -----------------
normalizer = Normalizer()
stopwords = set(stopwords_list())

try:
    with open("stop-words.txt", "r", encoding="utf-8") as f:
        custom_stopwords = {normalizer.normalize(line.strip()) for line in f if line.strip()}
        stopwords.update(custom_stopwords)
        stopwords.discard("روزه")
except FileNotFoundError:
    print("فایل stop-words.txt پیدا نشد، فقط استاپ‌وردهای hazm استفاده می‌شود.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

# ----------------- Helpers -----------------
def preprocess(text: str) -> str:
    text = normalizer.normalize(str(text))
    tokens = [t for t in word_tokenize(text) if t not in stopwords]
    return " ".join(tokens)

def process_text(text: str, cluster_center: np.ndarray, token_embedding_cache: dict) -> dict:
    tokens = [t for t in word_tokenize(normalizer.normalize(str(text))) if t not in stopwords]
    if not tokens:
        return {"متن سوال": text, "کلمات کلیدی": ""}

    token_embs = np.array([token_embedding_cache[t] for t in tokens], dtype=np.float32)
    similarities = token_embs @ cluster_center

    top_indices = similarities.argsort()[-TOP_KW:][::-1]
    keywords = [tokens[idx] for idx in top_indices]

    # remove duplicates, keep order
    seen = set()
    unique_tokens = [t for t in keywords if not (t in seen or seen.add(t))]

    return {"متن سوال": text, "کلمات کلیدی": ", ".join(unique_tokens)}

def process_chunk(chunk: pd.DataFrame) -> list[dict]:
    texts = chunk["متن سوال"].astype(str).tolist()

    processed_docs = [preprocess(text) for text in texts]

    embeddings = model.encode(
        processed_docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    n_clusters = min(max(2, len(texts) // 10), 50)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    # token embeddings cache (once per chunk)
    unique_tokens = list({
        t
        for text in texts
        for t in word_tokenize(normalizer.normalize(str(text)))
        if t not in stopwords
    })

    token_embedding_cache = {}
    if unique_tokens:
        token_embs = model.encode(
            unique_tokens,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        token_embedding_cache = dict(zip(unique_tokens, token_embs))

    def worker(i: int) -> dict:
        return process_text(
            texts[i],
            kmeans.cluster_centers_[cluster_labels[i]].astype(np.float32),
            token_embedding_cache
        )

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, range(len(texts))))

    return results

# ----------------- Main -----------------
if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    chunks = pd.read_csv(INPUT_FILE, usecols=["متن سوال"], chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} ...")
        results = process_chunk(chunk)
        df_out = pd.DataFrame(results)[["متن سوال", "کلمات کلیدی"]]

        df_out.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
            mode="a",
            header=(i == 0)
        )

    print("✅ keyword CSV ساخته شد:", OUTPUT_FILE)
import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

META = "keywords_output.csv"
OUT = "artifacts/kw_emb.npy"

os.makedirs("artifacts", exist_ok=True)

if os.path.exists(OUT):
    print("⚠ embeddings already exist. Skipping.")
    raise SystemExit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

df = pd.read_csv(META)
texts = df["کلمات کلیدی"].fillna("").astype(str).tolist()

emb = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=256,
    show_progress_bar=True
).astype("float32")

np.save(OUT, emb)
print("✅ embeddings saved:", OUT, "shape=", emb.shape)
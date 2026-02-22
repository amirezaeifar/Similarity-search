import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib

EMB = "artifacts/kw_emb.npy"
CLUSTER_IDS = "artifacts/cluster_ids.npy"
CENTROIDS = "artifacts/centroids.npy"
KMEANS_MODEL = "artifacts/kmeans.joblib"

if os.path.exists(CLUSTER_IDS) and os.path.exists(CENTROIDS):
    print("⚠ clustering already exists. Skipping.")
    raise SystemExit(0)

if not os.path.exists(EMB):
    raise FileNotFoundError(f"Embedding file not found: {EMB}")

emb = np.load(EMB, mmap_mode="r")
if emb.dtype != np.float32:
    emb = emb.astype(np.float32)

n_clusters = 2500

kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    batch_size=8192,
    random_state=42,
    reassignment_ratio=0.01
)

kmeans.fit(emb)

cluster_ids = kmeans.labels_.astype(np.int32)
centroids = kmeans.cluster_centers_.astype(np.float32)

# normalize centroids
centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

np.save(CLUSTER_IDS, cluster_ids)
np.save(CENTROIDS, centroids)
joblib.dump(kmeans, KMEANS_MODEL)

print("✅ clustering done.")
print("saved:", CLUSTER_IDS, CENTROIDS, KMEANS_MODEL)
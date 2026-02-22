import os
import numpy as np
import faiss
import pickle

EMB = "artifacts/kw_emb.npy"
CLUSTER_IDS = "artifacts/cluster_ids.npy"

OUT_INDEX = "artifacts/faiss_clusters"
OUT_MAP = "artifacts/faiss_maps"

os.makedirs(OUT_INDEX, exist_ok=True)
os.makedirs(OUT_MAP, exist_ok=True)

emb = np.load(EMB, mmap_mode="r")
cluster_ids = np.load(CLUSTER_IDS)
dim = emb.shape[1]

members = {}
for i, cid in enumerate(cluster_ids):
    members.setdefault(int(cid), []).append(i)

use_gpu = faiss.get_num_gpus() > 0
res = faiss.StandardGpuResources() if use_gpu else None

GPU_MIN_SIZE = 5000

for cid, ids in members.items():
    vecs = emb[np.array(ids, dtype=np.int64)].astype("float32")

    if use_gpu and len(ids) >= GPU_MIN_SIZE:
        cpu_index = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(vecs)
        index_to_save = faiss.index_gpu_to_cpu(gpu_index)
    else:
        index_to_save = faiss.IndexFlatIP(dim)
        index_to_save.add(vecs)

    faiss.write_index(index_to_save, f"{OUT_INDEX}/c_{cid}.index")
    with open(f"{OUT_MAP}/c_{cid}.pkl", "wb") as f:
        pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ FAISS per cluster built (Hybrid).")
# 🔍 Persian Semantic Duplicate Detection System

# 🔍 سیستم تشخیص سوالات تکراری معنایی فارسی

------------------------------------------------------------------------

## 🇬🇧 English Description

A scalable semantic duplicate detection system for Persian text using
SentenceTransformers, clustering, and FAISS.

This project detects semantically similar (duplicate) Persian questions
using a two-stage retrieval architecture designed for large-scale
datasets.

### 🚀 Features

-   Semantic similarity detection (not keyword-only matching)
-   Persian text normalization using **Hazm**
-   Custom + default stopword filtering
-   Scalable clustering with **MiniBatchKMeans**
-   Per-cluster FAISS vector indexing
-   CPU & GPU support (PyTorch + FAISS)
-   Two-stage retrieval pipeline
-   Production-ready Flask API

------------------------------------------------------------------------

## 🇮🇷 توضیحات فارسی

این پروژه یک سیستم مقیاس‌پذیر برای تشخیص سوالات تکراری به صورت معنایی در
زبان فارسی است.

در این سیستم از SentenceTransformers، خوشه‌بندی و FAISS برای جستجوی سریع
برداری استفاده شده تا بتوان سوالات مشابه معنایی را در دیتاست‌های بزرگ با
سرعت و دقت بالا شناسایی کرد.

### 🚀 ویژگی‌ها

-   تشخیص شباهت معنایی (نه صرفاً تطابق کلمه‌ای)
-   نرمال‌سازی متن فارسی با **Hazm**
-   استفاده از استاپ‌وردهای پیش‌فرض و سفارشی
-   خوشه‌بندی مقیاس‌پذیر با **MiniBatchKMeans**
-   ساخت ایندکس FAISS به ازای هر خوشه
-   پشتیبانی از CPU و GPU
-   معماری دو مرحله‌ای بازیابی
-   ارائه API آماده استفاده با Flask

------------------------------------------------------------------------

# 🧠 System Architecture \| معماری سیستم

## 1️⃣ Keyword Extraction & Preprocessing \| استخراج و پیش‌پردازش

-   Text normalization
-   Stopword removal
-   Keyword extraction

## 2️⃣ Embedding Generation \| تولید امبدینگ

Model: `paraphrase-multilingual-mpnet-base-v2`\
Embeddings normalized for cosine similarity.

## 3️⃣ Clustering \| خوشه‌بندی

-   MiniBatchKMeans
-   Normalized cluster centroids

## 4️⃣ FAISS Indexing \| ساخت ایندکس FAISS

-   One FAISS index per cluster
-   Hybrid CPU/GPU indexing

## 5️⃣ Two-Stage Retrieval \| بازیابی دو مرحله‌ای

**Stage 1:**\
Cluster routing + FAISS candidate retrieval

**Stage 2:**\
Full sentence embedding + semantic reranking

------------------------------------------------------------------------

# 📂 Project Structure

    .
    ├── 01_extract_keywords.py
    ├── 02_embed_keywords.py
    ├── 03_cluster_keywords.py
    ├── 04_build_faiss_clusters.py
    ├── 05_server.py
    ├── requirements.txt
    ├── stop-words.txt
    ├── artifacts/

------------------------------------------------------------------------

# ⚙️ Installation

``` bash
git clone <your-repo-url>
cd <project-folder>

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🏗 Build Pipeline

``` bash
python 01_extract_keywords.py
python 02_embed_keywords.py
python 03_cluster_keywords.py
python 04_build_faiss_clusters.py
```

------------------------------------------------------------------------

# 🌐 Run API

``` bash
python 05_server.py
```

Server:

    http://localhost:5005

------------------------------------------------------------------------

# 📡 API Example

### Request

``` json
{
  "question": "متن سوال جدید شما"
}
```

### Response

``` json
{
  "question": "...",
  "duplicates_count": 2,
  "best_match": {
    "sent_score": 0.91,
    "text": "متن سوال مشابه"
  }
}
```

------------------------------------------------------------------------

# 🛠 Tech Stack

-   Python
-   SentenceTransformers
-   PyTorch
-   FAISS
-   Scikit-learn
-   Hazm
-   Flask
-   NumPy
-   Pandas

------------------------------------------------------------------------

# 🎯 Use Cases \| موارد استفاده

-   Large-scale Persian Q&A datasets
-   Duplicate question detection
-   Semantic search systems
-   Intelligent support systems

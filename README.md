# 🏛️ Strata RAG: Global Archaeological AI

Strata is a locally-deployed, 7-billion parameter Retrieval-Augmented Generation (RAG) pipeline built for deep archaeological research. It utilizes a dynamically scraped ChromaDB vector space encompassing global encyclopedias and peer-reviewed academic journals.

## 📊 Performance & Benchmarks
Strata is built on the **Qwen2.5-7B-Instruct** architecture, optimized with 4-bit Unsloth quantization to run locally on consumer hardware. 

* **Peak VRAM Consumption:** ~7.0 GB / 8.0 GB (Tested on RTX 5060)
* **Time-to-First-Token (TTFT):** ~0.8s – 1.2s
* **Generation Throughput:** 38 – 45 tokens per second (t/s)
* **Context Processing:** Native 8,192 token window. 

## 🏗️ Architecture
* **Vector Database:** ChromaDB
* **Embedder:** `BAAI/bge-large-en-v1.5`
* **Inference Engine:** Unsloth `FastLanguageModel` (4-bit optimized)
* **Smart Router:** Custom Streamlit intent-router to bypass RAG during casual conversation.

## 🚀 Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Ingest Data: `python3 scraper.py`, `python3 deep_scraper.py`, `python3 ingest.py`, `python3 indexer.py`
3. Launch UI: `streamlit run app.py`

# 🏛️ Strata RAG: Global Archaeological AI

Strata is a locally-deployed, 7-billion parameter Retrieval-Augmented Generation (RAG) pipeline built for deep archaeological research. It utilizes a dynamically scraped ChromaDB vector space encompassing global encyclopedias and peer-reviewed academic journals.

## 🧠 Why We Chose Qwen2.5-7B-Instruct
For a domain as dense as archaeology—which requires parsing complex academic terminology, historical dates, and multi-lingual context—the foundation model must have exceptional reasoning capabilities. 
* **State-of-the-Art in 7B:** Qwen2.5-7B consistently outperforms competitors (like Llama-3-8B and Mistral-7B) in analytical reasoning (MMLU: 74.2), making it the perfect base for understanding structural histories.
* **Context Handling:** It natively supports up to 131k context length (configured to 8,192 for this project), allowing it to ingest multiple long-form academic abstracts simultaneously during RAG retrieval without degrading output quality.

## ⚡ Fine-Tuning Methodology: Unsloth & QLoRA
Training a 7-billion parameter model traditionally requires massive server racks, but Strata was successfully fine-tuned on a single consumer laptop GPU (NVIDIA RTX 5060, 8GB VRAM).
* **Unsloth:** We utilized the Unsloth library, which rewrites PyTorch's attention kernels to be up to 2x faster and use 70% less VRAM. This allowed us to train the model locally without Out-Of-Memory (OOM) crashes.
* **QLoRA (Quantized Low-Rank Adaptation):** Instead of rewriting all 7 billion parameters, we froze the base model in 4-bit precision (via BitsAndBytes) and trained only a tiny subset of "adapter" weights (the LoRA). This taught the model the specific tone and strict non-hallucination rules required for the Strata pipeline while keeping the footprint ultra-lightweight.

## 📚 The Dataset: Deep-Web Academic Scraping
Standard AI models hallucinate historical facts because they rely on generic internet data. Strata is grounded in pure, peer-reviewed science.
* **Automated Ingestion:** Custom Python crawlers (e.g., `deep_scraper.py`) interface directly with academic APIs like Crossref to rip XML data from global archaeological databases.
* **Content:** The vector database contains hundreds of chunked excavation reports, structural histories, and peer-reviewed abstracts covering sites from Göbekli Tepe to Mesoamerica. 

## 🔍 The RAG Pipeline & BAAI Embeddings
To ensure the LLM only answers based on factual data, we implemented a sophisticated RAG architecture.
* **Why BAAI (`bge-large-en-v1.5`):** We selected the BAAI embedding model because it ranks at the top of the Massive Text Embedding Benchmark (MTEB) for open-source models. It maps our text chunks into highly accurate 1024-dimensional vectors, ensuring the retriever grabs the exact paragraph needed to answer the user's query.
* **Vector Storage:** ChromaDB acts as the persistent, local disk-backed database for lightning-fast vector similarity searches.

## 📦 Portability with GGUF
While the model was trained in Safetensors, Strata supports exportation to the **GGUF** format (compatible with tools like Ollama and llama.cpp).
* **Hardware Agnostic:** GGUF allows the model to run on almost any hardware, dynamically splitting the workload between the CPU and GPU. 
* **High Compression:** GGUF quantizes the final merged weights into a single, highly compressed file, making it incredibly easy to share, deploy, and execute in production environments where VRAM is severely limited.

---

| Feature | Strata (Local RAG + LoRA) | Cloud APIs (GPT-4 / Claude) | Standard Local LLMs |
| :--- | :--- | :--- | :--- |
| **Data Privacy** | **100% Local (Air-Gapped)** | Data sent to corporate servers | 100% Local |
| **Cost** | **$0.00 (Free to run)** | High API token costs | $0.00 |
| **Domain Accuracy** | **Expert (Ground-truth RAG)** | General (Prone to hallucination) | General (Prone to hallucination) |
| **Hardware Required** | **Consumer GPU (8GB VRAM)** | None (Cloud) | Enterprise GPU (16GB+ VRAM) |
| **Latency** | **Sub-second TTFT** | Dependent on network speed | Varies heavily |

### The Strata Plus Points
1. **Smart Intent Routing:** Strata uses a custom regex router in its Streamlit UI. If you say "hello" or ask a general conversational question, it bypasses the heavy vector database for an instant reply. It only triggers the RAG pipeline for deep research queries.
2. **Absolute Data Sovereignty:** You control the knowledge base. By running the scraping scripts, you can instantly teach the model about a newly discovered excavation site without retraining the neural network.
3. **Refusal to Hallucinate:** If the vector database returns empty, the model is explicitly instructed to rely on its general knowledge but *transparently inform the user* that the data did not come from the official Strata database.

---

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

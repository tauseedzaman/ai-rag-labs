# 🧠 AI RAG Labs

A hands-on, experiment-driven repository for learning and engineering Retrieval-Augmented Generation (RAG) systems using:

* 🧩 **Chroma** (Vector Database)
* 🦙 **Ollama** (Local Embeddings + LLM)
* 📄 PDF ingestion pipeline
* 📊 Retrieval benchmarking with measurable metrics

This repo focuses on **retrieval engineering first**, before generation.

---

# 🚀 Philosophy

RAG is not:

> PDF → LLM → Answer

RAG is:

```text
Document → Structure → Chunk → Embed → Index → Retrieve → (Then Generate)
```

If retrieval is weak, generation cannot fix it.

This repo isolates and optimizes each stage step-by-step.

---


# 🏗 Repository Structure
[🧪 Lab 01 – PDF Ingestion with Metadata](lab01.md)

[🧪 Lab 02 – Chunking Strategy Benchmark](lab02.md)

[🧪 Lab 03 – Retrieval Evaluation Harness](lab03.md)


# ⚙️ Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Install embedding model in Ollama

```bash
ollama pull nomic-embed-text
```

### 3️⃣ Add a PDF

Place your test PDF in:

```
data/raw/sample.pdf
```

---

# 🧠 What This Repo Is Really About

This repository is about learning:

* Information Retrieval fundamentals
* Chunking tradeoffs
* Embedding behavior
* Ranking metrics
* Experimental isolation
* Evidence-based optimization

This is not just “build RAG app”.

This is:

> Engineering retrieval systems correctly.

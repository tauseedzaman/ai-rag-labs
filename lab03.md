# 🧪 Lab 03 — Retrieval Evaluation Harness (hit@k, MRR)

## 🎯 Goal
Build a repeatable evaluation harness for retrieval quality using real, curated queries (not synthetic).

This enables regression testing whenever we change:
- chunking config
- embedding model
- retrieval strategy (hybrid, reranking, etc.)

---

## ✅ What We Evaluate
Given a query, retrieve top-k chunks from Chroma and check if any retrieved chunk matches the ground truth.

Ground truth is defined by:
- `gold_pages`: list of correct page numbers
- optionally `gold_sources`: list of valid sources (PDF name)

---

## 📁 Eval Dataset Format (JSONL)

File: `data/eval/lab03_eval.jsonl`

Each line:
```json
{
  "id": "q01",
  "query": "How much is PIF presale price?",
  "gold_pages": [5],
  "gold_sources": ["sample_pdf"],
  "notes": "..."
}
```

## 📊 Metrics

### hit@k

A query is a "hit" if any correct page appears in the top-k retrieved results.

### MRR@k (Mean Reciprocal Rank)

Rewards higher placement of the first correct result.

* rank 1 → 1.0
* rank 2 → 0.5
* rank 5 → 0.2
* not found → 0.0

MRR is averaged across all queries.

---

## 🏗 How It Works (Pipeline)

1. Load eval queries from JSONL
2. Embed each query with Ollama embeddings
3. Retrieve top-k chunks from Chroma
4. Mark retrieval as correct if `page ∈ gold_pages` (and source matches if provided)
5. Compute hit@k and MRR@k
6. Save a detailed report JSON for debugging

---

## ▶️ Run

```bash
python src/labs/lab03_eval_harness.py
```

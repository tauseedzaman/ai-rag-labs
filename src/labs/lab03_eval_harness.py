import os
import json
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
import chromadb
import ollama
from tqdm import tqdm

load_dotenv()


# -----------------------------
# 1) Data loading
# -----------------------------
def load_evalset(eval_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation queries from a JSONL file.
    Each line must be a JSON object.
    Required keys: id, query, gold_pages
    Optional keys: gold_sources, notes
    """
    if not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"Eval file not found: {eval_path}\n"
            f"Create it first (example below in lab03.md)."
        )

    rows: List[Dict[str, Any]] = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # minimal validation
            for key in ["id", "query", "gold_pages"]:
                if key not in obj:
                    raise ValueError(f"Missing '{key}' on line {ln}")

            if not isinstance(obj["gold_pages"], list) or not obj["gold_pages"]:
                raise ValueError(f"'gold_pages' must be a non-empty list on line {ln}")

            obj.setdefault("gold_sources", [])
            obj.setdefault("notes", "")
            rows.append(obj)

    return rows


# -----------------------------
# 2) Embeddings + retrieval
# -----------------------------
def embed_text(text: str, embed_model: str) -> List[float]:
    """Generate an embedding for a query using Ollama."""
    r = ollama.embeddings(model=embed_model, prompt=text)
    return r["embedding"]


def query_collection(col, query_embedding: List[float], k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Query Chroma with the query embedding.
    Returns (metadatas, distances) for top-k results.
    """
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "distances"],
    )
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    return metadatas, distances


# -----------------------------
# 3) Relevance judging
# -----------------------------
def is_relevant(meta: Dict[str, Any], gold_pages: List[int], gold_sources: List[str]) -> bool:
    """
    Define what counts as a "correct retrieval".
    - Correct if page matches any gold page
    - If gold_sources provided, source must also match
    """
    page = meta.get("page", None)
    source = meta.get("source", None)

    page_ok = (page is not None) and (int(page) in [int(p) for p in gold_pages])

    if not page_ok:
        return False

    if gold_sources:
        return source in gold_sources

    return True


# -----------------------------
# 4) Scoring a single query
# -----------------------------
def score_query(retrieved_metas: List[Dict[str, Any]], gold_pages: List[int], gold_sources: List[str], k: int) -> Dict[str, Any]:
    """
    Returns:
    - hit: 1 if relevant doc appears in top-k, else 0
    - rr: reciprocal rank (1/rank) of first relevant, else 0
    - first_rank: rank of first relevant (1-indexed) or None
    """
    first_rank: Optional[int] = None

    for idx, meta in enumerate(retrieved_metas[:k], start=1):
        if is_relevant(meta, gold_pages, gold_sources):
            first_rank = idx
            break

    hit = 1 if first_rank is not None else 0
    rr = (1.0 / first_rank) if first_rank is not None else 0.0

    return {"hit": hit, "rr": rr, "first_rank": first_rank}


# -----------------------------
# 5) Full evaluation loop
# -----------------------------
def evaluate(evalset: List[Dict[str, Any]], col, embed_model: str, k: int = 5) -> Dict[str, Any]:
    """
    Evaluate retrieval over all queries.
    Returns summary + per-query breakdown.
    """
    hits = 0
    rr_sum = 0.0
    per_query = []

    for row in tqdm(evalset, desc="Evaluating"):
        qid = row["id"]
        query = row["query"]
        gold_pages = row["gold_pages"]
        gold_sources = row.get("gold_sources", [])

        q_emb = embed_text(query, embed_model=embed_model)
        metas, dists = query_collection(col, q_emb, k=k)

        score = score_query(metas, gold_pages, gold_sources, k=k)
        hits += score["hit"]
        rr_sum += score["rr"]

        per_query.append({
            "id": qid,
            "query": query,
            "gold_pages": gold_pages,
            "gold_sources": gold_sources,
            "hit": score["hit"],
            "first_rank": score["first_rank"],
            "rr": round(score["rr"], 4),
            "top_pages": [m.get("page") for m in metas],
            "top_sources": [m.get("source") for m in metas],
            "top_distances": [round(float(d), 4) for d in dists],
        })

    n = max(len(evalset), 1)
    hit_at_k = hits / n
    mrr_at_k = rr_sum / n

    return {
        "k": k,
        "hit_at_k": round(hit_at_k, 4),
        "mrr_at_k": round(mrr_at_k, 4),
        "num_queries": len(evalset),
        "per_query": per_query,
    }


def main():
    # env config
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ["OLLAMA_HOST"] = ollama_base

    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")

    collection_name = os.getenv("COLLECTION_NAME", "lab01_pdf")

    eval_path = os.getenv("EVAL_PATH", "data/eval/lab03_eval.jsonl")
    k = int(os.getenv("TOP_K", "5"))

    # load eval set
    evalset = load_evalset(eval_path)

    # connect to chroma
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_collection(name=collection_name)

    # run evaluation
    report = evaluate(evalset, col, embed_model=embed_model, k=k)

    print("\n=== Lab 03 Results ===")
    print(f"Collection: {collection_name}")
    print(f"Embed model: {embed_model}")
    print(f"Queries: {report['num_queries']}")
    print(f"hit@{k}: {report['hit_at_k']}")
    print(f"MRR@{k}: {report['mrr_at_k']}")

    # show failures for debugging
    failures = [r for r in report["per_query"] if r["hit"] == 0]
    if failures:
        print("\n--- Failures (hit=0) ---")
        for r in failures[:10]:
            print(f"- {r['id']}: top_pages={r['top_pages']} | query={r['query'][:90]}...")
    else:
        print("\nNo failures 🎉")

    # optional: save report
    out_path = os.getenv("REPORT_PATH", "data/eval/lab03_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
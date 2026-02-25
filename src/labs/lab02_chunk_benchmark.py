import os
import re
import json
import uuid
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm
import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


@dataclass
class PageDoc:
    text: str
    metadata: Dict[str, Any]


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def read_pdf_pages(pdf_path: str, source_name: str) -> List[PageDoc]:
    reader = PdfReader(pdf_path)
    pages: List[PageDoc] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        pages.append(
            PageDoc(
                text=text,
                metadata={
                    "source": source_name,
                    "file_path": pdf_path,
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                },
            )
        )
    return pages


def chunk_pages(pages: List[PageDoc], chunk_size: int, chunk_overlap: int) -> List[PageDoc]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[PageDoc] = []
    for p in pages:
        if len(p.text) < 80:
            continue
        splits = splitter.split_text(p.text)
        for idx, t in enumerate(splits):
            meta = dict(p.metadata)
            meta["chunk_index"] = idx
            meta["chunk_id"] = str(uuid.uuid4())
            meta["chunk_chars"] = len(t)
            chunks.append(PageDoc(text=t, metadata=meta))
    return chunks


def embed_texts(texts: List[str], embed_model: str) -> List[List[float]]:
    vectors: List[List[float]] = []
    for t in tqdm(texts, desc="Embedding", leave=False):
        r = ollama.embeddings(model=embed_model, prompt=t)
        vectors.append(r["embedding"])
    return vectors


def sentences(text: str) -> List[str]:
    # Simple sentence split good enough for lab purposes
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 40]


def make_synthetic_eval(pages: List[PageDoc], n: int = 20, seed: int = 7) -> List[Dict[str, Any]]:
    """
    Create evaluation queries from the PDF itself:
    - Pick random pages
    - Pick a medium-length sentence
    - Use that as a query
    - Gold label = that page number
    This is not a perfect real-world eval, but it *is* great for comparing chunk configs.
    """
    random.seed(seed)

    candidates = []
    for p in pages:
        sents = sentences(p.text)
        if not sents:
            continue
        # prefer medium sentences
        sents = [s for s in sents if 60 <= len(s) <= 180] or sents
        candidates.append((p.metadata["page"], random.choice(sents)))

    random.shuffle(candidates)
    picked = candidates[:n]

    evalset = []
    for page_num, sent in picked:
        # Slightly “query-ify” by trimming to first ~120 chars
        q = sent
        if len(q) > 140:
            q = q[:140].rsplit(" ", 1)[0]
        evalset.append({"query": q, "gold_page": page_num})
    return evalset


def load_or_create_eval(eval_path: str, pages: List[PageDoc]) -> List[Dict[str, Any]]:
    if os.path.exists(eval_path):
        rows = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        # expected schema: {"query": "...", "gold_page": 12}
        return rows

    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    evalset = make_synthetic_eval(pages, n=20)
    with open(eval_path, "w", encoding="utf-8") as f:
        for row in evalset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return evalset


def evaluate_collection(col, evalset: List[Dict[str, Any]], embed_model: str, k: int = 5) -> Tuple[float, float]:
    """
    Returns (hit_at_k, mrr)
    hit@k: fraction of queries where gold_page appears in top k retrieved results
    mrr: mean reciprocal rank of first correct page
    """
    hits = 0
    rr_sum = 0.0

    for row in tqdm(evalset, desc="Eval queries", leave=False):
        q = row["query"]
        gold_page = int(row["gold_page"])

        q_emb = ollama.embeddings(model=embed_model, prompt=q)["embedding"]
        res = col.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["metadatas", "distances"],
        )

        metas = res["metadatas"][0]
        found_rank = None
        for i, m in enumerate(metas, start=1):
            if int(m.get("page", -1)) == gold_page:
                found_rank = i
                break

        if found_rank is not None:
            hits += 1
            rr_sum += 1.0 / found_rank

    hit_at_k = hits / max(len(evalset), 1)
    mrr = rr_sum / max(len(evalset), 1)
    return hit_at_k, mrr


def main():
    pdf_path = os.getenv("PDF_PATH", "data/raw/PIF-Whitepaper.pdf")
    source_name = os.getenv("SOURCE_NAME", "sample_pdf")

    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    bench_collection_prefix = os.getenv("BENCH_COLLECTION_PREFIX", "lab02_bench")
    eval_path = os.getenv("EVAL_PATH", "data/eval/lab02_eval.jsonl")

    os.environ["OLLAMA_HOST"] = ollama_base

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Reading PDF: {pdf_path}")
    pages = read_pdf_pages(pdf_path, source_name)
    print(f"Pages: {len(pages)}")

    evalset = load_or_create_eval(eval_path, pages)
    print(f"Eval queries: {len(evalset)} (from {eval_path})")

    # Grid to benchmark
    grid = [
        (400, 50),
        (400, 150),
        (800, 150),
        (800, 250),
        (1200, 150),
        (1200, 250),
    ]

    client = chromadb.PersistentClient(path=chroma_dir)

    results = []
    for chunk_size, overlap in grid:
        collection_name = f"{bench_collection_prefix}_cs{chunk_size}_ov{overlap}"

        # start clean each time (so results are comparable)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        print(f"\n=== Benchmark: chunk_size={chunk_size} overlap={overlap} ===")
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=overlap)
        print(f"Chunks: {len(chunks)}")

        texts = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]
        ids = [c.metadata["chunk_id"] for c in chunks]

        embeddings = embed_texts(texts, embed_model=embed_model)

        col = client.get_or_create_collection(collection_name)
        batch = 200
        for i in tqdm(range(0, len(texts), batch), desc="Upserting", leave=False):
            col.upsert(
                ids=ids[i : i + batch],
                documents=texts[i : i + batch],
                metadatas=metas[i : i + batch],
                embeddings=embeddings[i : i + batch],
            )

        hit5, mrr5 = evaluate_collection(col, evalset, embed_model=embed_model, k=5)
        results.append((chunk_size, overlap, hit5, mrr5))

        print(f"hit@5={hit5:.3f} | MRR@5={mrr5:.3f}")

    # sort by MRR first, then hit@5
    results.sort(key=lambda x: (x[3], x[2]), reverse=True)

    print("\n\n=== Results (sorted best → worst) ===")
    print("chunk_size  overlap   hit@5   mrr@5")
    for cs, ov, h, m in results:
        print(f"{cs:<10} {ov:<8} {h:<6.3f} {m:<6.3f}")

    best = results[0]
    print(
        f"\nBest config (by mrr@5): chunk_size={best[0]}, overlap={best[1]} "
        f"(hit@5={best[2]:.3f}, mrr@5={best[3]:.3f})"
    )


if __name__ == "__main__":
    main()
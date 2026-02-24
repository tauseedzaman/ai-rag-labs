import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any

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
    """Light cleaning: normalize whitespace + remove repeated blank lines."""
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
        # store even short pages? we’ll filter later
        pages.append(
            PageDoc(
                text=text,
                metadata={
                    "source": source_name,
                    "file_path": pdf_path,
                    "page": i + 1,  # 1-indexed for humans
                    "total_pages": len(reader.pages),
                },
            )
        )

    return pages


def chunk_pages(pages: List[PageDoc], chunk_size: int = 900, chunk_overlap: int = 150) -> List[PageDoc]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[PageDoc] = []
    for p in pages:
        if len(p.text) < 40:
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
    """Embed via Ollama embeddings API."""
    vectors: List[List[float]] = []
    for t in tqdm(texts, desc="Embedding"):
        r = ollama.embeddings(model=embed_model, prompt=t)
        vectors.append(r["embedding"])
    return vectors


def main():
    pdf_path = os.getenv("PDF_PATH", "data/raw/PIF-Whitepaper.pdf")
    source_name = os.getenv("SOURCE_NAME", "sample_pdf")

    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "lab01_pdf")

    # point ollama client to your base URL if needed
    # ollama python library reads OLLAMA_HOST too, but we’ll set env for clarity
    os.environ["OLLAMA_HOST"] = ollama_base

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[1/5] Reading PDF pages: {pdf_path}")
    pages = read_pdf_pages(pdf_path, source_name)
    print(f"  Pages read: {len(pages)}")

    print("[2/5] Chunking pages")
    chunks = chunk_pages(pages)
    print(f"  Chunks created: {len(chunks)}")

    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [c.metadata["chunk_id"] for c in chunks]

    print("[3/5] Embedding chunks with Ollama")
    embeddings = embed_texts(texts, embed_model=embed_model)

    print("[4/5] Writing to Chroma")
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_or_create_collection(name=collection_name)

    # Upsert in batches to avoid big memory spikes
    batch = 100
    for i in tqdm(range(0, len(texts), batch), desc="Upserting"):
        col.upsert(
            ids=ids[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
            embeddings=embeddings[i : i + batch],
        )

    print(f"[5/5] Done. Chroma saved at: {chroma_dir} | collection: {collection_name}")

    # Quick sanity query
    q = "What is this document about?"
    print("\nSanity query:", q)
    q_emb = ollama.embeddings(model=embed_model, prompt=q)["embedding"]
    res = col.query(query_embeddings=[q_emb], n_results=3, include=["documents", "metadatas", "distances"])
    for rank in range(3):
        doc = res["documents"][0][rank]
        meta = res["metadatas"][0][rank]
        dist = res["distances"][0][rank]
        print(f"\n#{rank+1} distance={dist:.4f} page={meta.get('page')} chunk={meta.get('chunk_index')} source={meta.get('source')}")
        print(doc[:400], "...")


if __name__ == "__main__":
    main()
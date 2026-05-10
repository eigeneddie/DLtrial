"""
ingest_thermal_papers.py — Chilli Chiplets Knowledge Base Builder
=================================================================
Uses Docling to parse thermal engineering PDFs into Markdown, then
chunks and embeds them into a local Qdrant vector store.

Run once before launching the app:
    python rag/ingest_thermal_papers.py

Requires Ollama running locally with nomic-embed-text pulled:
    ollama pull nomic-embed-text
"""

import re
import sys
from pathlib import Path
from typing import Iterable

# ── Docling ──────────────────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter

# ── Vector DB ────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_community.embeddings import OllamaEmbeddings
from tqdm import tqdm

# ── Local config ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ── Chunking (same logic as data/rag_app/ingest.py) ─────────────────────────

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> Iterable[str]:
    """
    Section-aware chunking for engineering text.
    Splits on Markdown headings first, then paragraphs for oversized blocks.
    Adds trailing overlap between consecutive chunks for retrieval continuity.
    """
    clean = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not clean:
        return

    sections = re.split(r"\n(?=#{1,6}\s)", clean)
    normalized = [s.strip() for s in sections if s.strip()] or [clean]

    assembled: list[str] = []
    current = ""
    for sec in normalized:
        if len(sec) > chunk_size * 1.5:
            for para in re.split(r"\n{2,}", sec):
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) + 2 <= chunk_size:
                    current = (current + "\n\n" + para).strip()
                else:
                    if current:
                        assembled.append(current)
                    current = para
            continue
        if len(current) + len(sec) + 2 <= chunk_size:
            current = (current + "\n\n" + sec).strip()
        else:
            if current:
                assembled.append(current)
            current = sec
    if current:
        assembled.append(current)

    for i, ch in enumerate(assembled):
        if i == 0 or overlap <= 0:
            yield ch
            continue
        prev = assembled[i - 1]
        tail = prev[-overlap:] if len(prev) > overlap else prev
        yield (tail + "\n\n" + ch).strip()[:1500]


# ── Ingestion ─────────────────────────────────────────────────────────────────

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """Use Docling to convert a PDF to clean Markdown text."""
    print(f"  📄 Parsing: {Path(pdf_path).name}")
    converter = DocumentConverter()
    result    = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def ingest_all(pdf_paths: list[str]):
    print("=" * 60)
    print("  Chilli Chiplets — Knowledge Base Ingestion")
    print("=" * 60)

    # Connect to local Qdrant
    client = QdrantClient(path=config.QDRANT_PATH)

    # Re-create collection (clean slate)
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config=VectorParams(size=config.VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"  ✅ Created Qdrant collection: '{config.COLLECTION}'")

    # Load embeddings model via Ollama
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    print(f"  🔢 Embedding model: {config.EMBEDDING_MODEL}")

    points   = []
    point_id = 0

    for pdf_path in pdf_paths:
        if not Path(pdf_path).exists():
            print(f"  ⚠️  Skipping missing file: {pdf_path}")
            continue

        # Parse PDF → Markdown via Docling
        markdown = parse_pdf_to_markdown(pdf_path)
        stem     = Path(pdf_path).stem

        # Chunk and embed
        chunks = list(chunk_text(markdown))
        print(f"     → {len(chunks)} chunks")

        for idx, chunk in enumerate(tqdm(chunks, desc=f"     Embedding {stem}", leave=False)):
            vec = embeddings.embed_query(chunk)
            points.append(PointStruct(
                id      = point_id,
                vector  = vec,
                payload = {
                    "id"          : f"{stem}::chunk-{idx+1}",
                    "title"       : stem,
                    "source_file" : pdf_path,
                    "source_type" : "thermal_paper",
                    "chunk_index" : idx + 1,
                    "content"     : chunk,
                },
            ))
            point_id += 1

    # Upsert all points
    if points:
        client.upsert(collection_name=config.COLLECTION, points=points)

    print()
    print(f"  ✅ Ingestion complete — {point_id} chunks indexed into '{config.COLLECTION}'")
    print(f"  📁 Qdrant storage: {config.QDRANT_PATH}")
    print()
    print("  Next step: streamlit run app.py")


if __name__ == "__main__":
    ingest_all(config.THERMAL_PDF_PATHS)

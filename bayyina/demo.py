
from pathlib import Path
from bayyina.chunking import chunk_text
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever
from bayyina.cite import format_citation

chunks = []

for p in sorted(Path("data/contracts").glob("*.txt")):
    chunks += chunk_text(
        p.stem,
        p.read_text(encoding="utf-8"),
        500,
        50
    )

retriever = HybridRetriever(chunks, HashingEmbedder())

query = "متى يجوز إنهاء عقد التوريد بإخطار كتابي؟"

for hit in retriever.search(query, top_k=3):
    print(format_citation(hit))

from app.pipeline.builder import build_pipeline
from app.ingestion.document_loader import (
    load_documents
)

from app.ingestion.chunker import (
    create_chunks
)

from app.retrieval.bm25_retriever import (
    BM25Retriever
)
config = build_pipeline(
    "dense_minilm"
)

documents = load_documents()

all_chunks = []

for document in documents:
    chunks = create_chunks(
        document["text"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    all_chunks.extend(chunks)

retriever = BM25Retriever(
    all_chunks
)

results = retriever.retrieve(
    "What are the responsibilities of the contractor?",
    k=3
)

for i, result in enumerate(results, start=1):

    print("\n" + "=" * 50)

    print(f"Result {i}")

    print(result[:500])
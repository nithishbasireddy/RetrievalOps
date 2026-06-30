from app.pipeline.builder import (
    build_pipeline as build_pipeline_config
)
from app.ingestion.document_loader import load_documents
from app.ingestion.chunker import create_chunks
from app.embeddings.embedding_manager import get_embedding_model
from app.retrieval.vector_store import create_vectorstore


def build_index(config_name):

    config = build_pipeline_config(config_name)

    print(f"\nBuilding Index: {config_name}")

    documents = load_documents()

    all_chunks = []

    for document in documents:

        chunks = create_chunks(
            document["text"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )

        all_chunks.extend(chunks)

    print(f"Documents Loaded : {len(documents)}")
    print(f"Chunks Created   : {len(all_chunks)}")

    embeddings = get_embedding_model(
        config["embedding_model"]
    )

    vectorstore = create_vectorstore(
        chunks=all_chunks,
        embeddings=embeddings,
        persist_directory=f"data/vectorstores/{config_name}"
    )

    print("Index built successfully.")

    return vectorstore


if __name__ == "__main__":

    build_index("dense_minilm")
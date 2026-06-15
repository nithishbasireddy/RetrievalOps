from app.pipeline.builder import (
    build_pipeline as build_pipeline_config
)
from app.ingestion.document_loader import load_documents
from app.ingestion.chunker import create_chunks
from app.embeddings.embedding_manager import get_embedding_model
from app.retrieval.vector_store import create_vectorstore
from app.retrieval.retriever import retrieve
from app.pipeline.query_runner import run_query


def run_pipeline(config_name, query):

    config = build_pipeline_config(config_name)

    documents = load_documents()

    total_chunks = 0
    all_chunks = []

    for document in documents:

        chunks = create_chunks(
            document["text"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )

        all_chunks.extend(chunks)
        total_chunks += len(chunks)

    embeddings = get_embedding_model(
        config["embedding_model"]
    )

    

    vectorstore = create_vectorstore(
        chunks=all_chunks,
        embeddings=embeddings,
        persist_directory=f"data/vectorstores/{config_name}"
    )

    results = retrieve(
        vectorstore=vectorstore,
        query=query,
        k=config["top_k"]
    )

    return results

def build_pipeline(config_name):

    config = build_pipeline_config(
        config_name
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

    embeddings = get_embedding_model(
        config["embedding_model"]
    )

    vectorstore = create_vectorstore(
        chunks=all_chunks,
        embeddings=embeddings,
        persist_directory=f"data/vectorstores/{config_name}"
    )

    return {
        "config": config,
        "vectorstore": vectorstore
    }

def main():

    query = "What are the responsibilities of the contractor?"

    pipeline = build_pipeline(
        "dense_minilm"
    )

    results = run_query(
        vectorstore=pipeline["vectorstore"],
        query=query,
        top_k=pipeline["config"]["top_k"]
    )

    print("\nQUERY:")
    print(query)

    print("\nTOP RESULTS:\n")

    for i, doc in enumerate(results, start=1):

        print("=" * 50)

        print(f"Result {i}")

        print(doc.page_content[:500])

        print()


if __name__ == "__main__":
    main()
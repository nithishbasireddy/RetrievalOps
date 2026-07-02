from app.pipeline.builder import (
    build_pipeline as build_pipeline_config
)

from app.embeddings.embedding_manager import (
    get_embedding_model
)

from app.retrieval.vector_store import (
    load_vectorstore
)

from app.retrieval.retriever import retrieve


def run_query(
    config_name,
    query
):

    config = build_pipeline_config(
        config_name
    )

    embeddings = get_embedding_model(
        config["embedding_model"]
    )

    vectorstore = load_vectorstore(
        embeddings=embeddings,
        persist_directory=f"data/vectorstores/{config_name}"
    )

    results = retrieve(
        vectorstore=vectorstore,
        query=query,
        k=config["top_k"]
    )

    return results
from app.retrieval.retriever import retrieve


def run_query(
    vectorstore,
    query,
    top_k
):

    results = retrieve(
        vectorstore=vectorstore,
        query=query,
        k=top_k
    )

    return results
def retrieve(
    vectorstore,
    query,
    k
):

    return vectorstore.similarity_search(
        query,
        k=k
    )
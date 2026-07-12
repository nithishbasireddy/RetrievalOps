from sentence_transformers import CrossEncoder
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

_RERANKER_CACHE = {}


def get_reranker_model():

    model_name = "BAAI/bge-reranker-v2-m3"

    if model_name not in _RERANKER_CACHE:

        print(f"\nLoading CrossEncoder: {model_name}")

        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)

    return _RERANKER_CACHE[model_name]


def get_all_chunks_from_vectorstore(vectorstore):

    data = vectorstore.get()

    chunks = []

    for text, metadata in zip(data["documents"], data["metadatas"]):

        chunks.append({
            "chunk_id": metadata["chunk_id"],
            "document": metadata["document"],
            "chunk_index": metadata["chunk_index"],
            "text": text
        })

    return chunks


def retrieve(
    vectorstore,
    query,
    k,
    retrieval_type="dense"
):

    if retrieval_type == "dense":

        return vectorstore.similarity_search(query, k=k)

    chunks = get_all_chunks_from_vectorstore(vectorstore)

    if not chunks:

        return []

    if retrieval_type == "bm25":

        retriever = BM25Retriever(chunks)

        return retriever.retrieve(query, k=k)

    elif retrieval_type == "hybrid":

        overfetch_k = min(50, len(chunks))

        dense_results = vectorstore.similarity_search(query, k=overfetch_k)

        bm25_retriever = BM25Retriever(chunks)

        bm25_results = bm25_retriever.retrieve(query, k=overfetch_k)

        hybrid_retriever = HybridRetriever(dense_results, bm25_results, k=60)

        return hybrid_retriever.retrieve(final_k=k)

    elif retrieval_type == "hybrid_rerank":

        overfetch_k = min(50, len(chunks))

        dense_results = vectorstore.similarity_search(query, k=overfetch_k)

        bm25_retriever = BM25Retriever(chunks)

        bm25_results = bm25_retriever.retrieve(query, k=overfetch_k)

        hybrid_retriever = HybridRetriever(dense_results, bm25_results, k=60)

        fused_results = hybrid_retriever.retrieve(final_k=overfetch_k)

        if not fused_results:

            return []

        reranker = get_reranker_model()

        pairs = [[query, doc.page_content] for doc in fused_results]

        scores = reranker.predict(pairs)

        ranked_results = [
            doc for _, doc in sorted(
                zip(scores, fused_results),
                key=lambda x: x[0],
                reverse=True
            )
        ]

        return ranked_results[:k]

    else:

        raise ValueError(f"Unknown retrieval type: {retrieval_type}")
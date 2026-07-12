from collections import defaultdict


class HybridRetriever:

    def __init__(
        self,
        dense_results,
        bm25_results,
        k=60
    ):

        self.dense_results = dense_results
        self.bm25_results = bm25_results
        self.k = k

    def retrieve(self, final_k=5):

        scores = defaultdict(float)

        documents = {}

        for rank, doc in enumerate(
            self.dense_results,
            start=1
        ):

            chunk_id = doc.metadata["chunk_id"]

            documents[chunk_id] = doc

            scores[chunk_id] += 1 / (
                self.k + rank
            )

        for rank, doc in enumerate(
            self.bm25_results,
            start=1
        ):

            chunk_id = doc.metadata["chunk_id"]

            documents[chunk_id] = doc

            scores[chunk_id] += 1 / (
                self.k + rank
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for chunk_id, _ in ranked[:final_k]:

            if chunk_id in documents:

                results.append(
                    documents[chunk_id]
                )

        return results
        
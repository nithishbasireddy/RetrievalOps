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

    def retrieve(self):

        scores = defaultdict(float)

        documents = {}

        for rank, doc in enumerate(
            self.dense_results,
            start=1
        ):

            text = doc.page_content

            documents[text] = doc

            scores[text] += 1 / (
                self.k + rank
            )

        for rank, text in enumerate(
            self.bm25_results,
            start=1
        ):

            scores[text] += 1 / (
                self.k + rank
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for text, _ in ranked:

            if text in documents:

                results.append(
                    documents[text]
                )

        return results
        
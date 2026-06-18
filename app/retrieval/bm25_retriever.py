from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        tokenized_chunks = [
            chunk.split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_chunks
        )

    def retrieve(
        self,
        query,
        k=3
    ):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for idx in ranked_indices[:k]:

            results.append(
                self.chunks[idx]
            )

        return results
import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        tokenized_chunks = [
            self.preprocess(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_chunks
        )

    def preprocess(self, text):

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        return text.split()

    def retrieve(
        self,
        query,
        k=3
    ):

        tokenized_query = self.preprocess(
            query
        )

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
            chunk = self.chunks[idx]
            results.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "document": chunk["document"],
                        "chunk_index": chunk["chunk_index"]
                    }
                )
            )

        return results
from app.pipeline.query_runner import run_query


def main():

    query = "What are the responsibilities of the contractor?"

    configs = [
        "dense_minilm",
        "dense_e5",
        "bm25",
        "hybrid_minilm",
        "hybrid_e5",
        "hybrid_rerank"
    ]

    for config in configs:

        print("\n" + "=" * 60)
        print(f"Config: {config}")
        print("=" * 60)

        try:

            results = run_query(config, query)

            print(f"Retrieved {len(results)} results:")

            for i, doc in enumerate(results, start=1):

                chunk_id = doc.metadata.get("chunk_id", "N/A")

                filename = doc.metadata.get("document", "N/A")

                print(f"\n[Result {i}] ID: {chunk_id} | File: {filename}")

                print(doc.page_content[:250] + "...")

        except Exception as e:

            print(f"Error: {e}")


if __name__ == "__main__":
    main()
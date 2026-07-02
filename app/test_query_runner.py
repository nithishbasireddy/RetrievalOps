from app.pipeline.query_runner import run_query


def main():

    query = "What are the responsibilities of the contractor?"

    results = run_query(
        config_name="dense_minilm",
        query=query
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
from app.benchmark.dataset_loader import (
    load_benchmark_dataset
)

from app.pipeline.pipeline_runner import (
    build_pipeline
)

from app.pipeline.query_runner import (
    run_query
)


def main():

    dataset = load_benchmark_dataset()

    print(
        f"\nRunning {len(dataset)} benchmark questions\n"
    )

    pipeline = build_pipeline(
        "dense_minilm"
    )

    for item in dataset:

        question = item["question"]

        print("=" * 70)

        print(f"Question {item['id']}")
        print(question)

        results = run_query(
            vectorstore=pipeline["vectorstore"],
            query=question,
            top_k=pipeline["config"]["top_k"]
        )

        print("\nTop Result:\n")

        print(
            results[0].page_content[:300]
        )

        print("\n")


if __name__ == "__main__":
    main()
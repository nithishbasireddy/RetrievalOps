from app.benchmark.dataset_loader import (
    load_benchmark_dataset
)

from app.pipeline.pipeline_runner import (
    run_pipeline
)


def main():

    dataset = load_benchmark_dataset()

    print(
        f"\nRunning {len(dataset)} benchmark questions\n"
    )

    for item in dataset:

        question = item["question"]

        print("=" * 70)

        print(f"Question {item['id']}")
        print(question)

        results = run_pipeline(
            "dense_minilm",
            question
        )

        print("\nTop Result:\n")

        print(
            results[0].page_content[:300]
        )

        print("\n")


if __name__ == "__main__":
    main()
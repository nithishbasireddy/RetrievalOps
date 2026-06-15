from app.benchmark.dataset_loader import (
    load_benchmark_dataset
)
from app.pipeline.pipeline_runner import (
    build_pipeline
)
from app.pipeline.query_runner import (
    run_query
)

CONFIGS = [
    "dense_minilm",
    "dense_e5"
]


def main():

    dataset = load_benchmark_dataset()

    print(
        f"\nRunning {len(dataset)} benchmark questions\n"
    )

    pipelines = {}

    for config_name in CONFIGS:

        print(f"\nRunning: {config_name}")

        pipeline = build_pipeline(
            config_name
        )

        pipelines[config_name] = pipeline

    for item in dataset:

        question = item["question"]

        for config_name in CONFIGS:

            pipeline = pipelines[config_name]

            results = run_query(
                vectorstore=pipeline["vectorstore"],
                query=question,
                top_k=pipeline["config"]["top_k"]
            )

            config_label = "MiniLM" if "minilm" in config_name else "E5"

            print("=" * 70)
            print(f"Question {item['id']}")
            print(question)

            print(f"\n{config_label} Result:\n")

            print(
                results[0].page_content[:300]
            )

            print("\n")


if __name__ == "__main__":
    main()
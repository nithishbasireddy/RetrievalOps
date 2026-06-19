from app.pipeline.pipeline_runner import build_pipeline
from app.benchmark.dataset_loader import load_benchmark_dataset
from app.pipeline.query_runner import run_query

dataset = load_benchmark_dataset()

configs = [
    "dense_minilm",
    "dense_e5"
]

for config_name in configs:

    print("\n" + "=" * 70)
    print(config_name.upper())
    print("=" * 70)

    pipeline = build_pipeline(
        config_name
    )

    for item in dataset[:3]:

        question = item["question"]

        results = run_query(
            vectorstore=pipeline["vectorstore"],
            query=question,
            top_k=1
        )

        print("\nQUESTION:")
        print(question)

        print("\nTOP RESULT:")

        print(
            results[0].page_content[:250]
        )

        print()
from app.benchmark.dataset_loader import (
    load_benchmark_dataset
)


def load_gold_dataset():

    dataset = load_benchmark_dataset()

    print()

    print(
        f"Gold Questions Loaded: "
        f"{len(dataset)}"
    )

    print()

    print(
        dataset[0]["question"]
    )

    print()

    print(
        dataset[0]["reference_answer"]
    )


if __name__ == "__main__":

    load_gold_dataset()
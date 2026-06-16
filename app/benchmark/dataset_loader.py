import json
from pathlib import Path


DATASET_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "benchmark_dataset"
    / "legal_benchmark_gold.json"
)


def load_benchmark_dataset():

    with open(DATASET_PATH, "r") as file:

        dataset = json.load(file)

    return dataset


if __name__ == "__main__":

    dataset = load_benchmark_dataset()

    print(
        f"Questions Loaded: "
        f"{len(dataset)}"
    )

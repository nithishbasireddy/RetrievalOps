from app.benchmark.dataset_loader import (
    load_benchmark_dataset
)

dataset = load_benchmark_dataset()

print()

print("Question:")
print(dataset[0]["question"])

print()

print("Reference Answer:")
print(dataset[0]["reference_answer"])
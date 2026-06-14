from app.pipeline.builder import build_pipeline


def main():

    config = build_pipeline(
        "dense_minilm"
    )

    print("\nRUNNING CONFIG\n")

    for key, value in config.items():

        print(f"{key}: {value}")

    print("\nPipeline execution will go here.")


if __name__ == "__main__":
    main()
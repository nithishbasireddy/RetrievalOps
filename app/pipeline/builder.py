from app.config.config_loader import load_configs


def build_pipeline(config_name):

    configs = load_configs()

    config = configs["configs"][config_name]

    return config


if __name__ == "__main__":

    pipeline = build_pipeline(
        "dense_minilm"
    )

    print("\nPIPELINE CONFIG:\n")

    for key, value in pipeline.items():

        print(f"{key}: {value}")
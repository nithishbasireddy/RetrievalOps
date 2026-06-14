import yaml

CONFIG_PATH = "app/config/experiment_configs.yaml"


def load_configs():

    print(f"Loading: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as file:

        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":

    configs = load_configs()

    print("\nCONFIGS LOADED:\n")

    print(configs)
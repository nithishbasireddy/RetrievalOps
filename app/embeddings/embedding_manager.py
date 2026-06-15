from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name):

    print(f"\nLoading Embedding Model:")
    print(model_name)

    return HuggingFaceEmbeddings(
        model_name=model_name
    )
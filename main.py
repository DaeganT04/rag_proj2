import logging
import warnings

from query import ask_question


def configure_logging():
    warnings.filterwarnings("ignore")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def print_result(result):
    print("\nAnswer:\n")
    print(result["answer"])
    print("\nRetrieved Chunks:\n")

    for chunk in result["chunks"]:
        print(f"Source: {chunk['source']} | Chunk: {chunk['chunk_id']}")
        print(chunk["text"])
        print("-" * 60)

    print()


def main():
    configure_logging()

    print("Custom RAG Assistant")
    print("Type 'quit' to exit.\n")

    while True:
        user_query = input("Ask a question: ").strip()

        if user_query.lower() == "quit":
            print("Goodbye.")
            break

        result = ask_question(user_query, n_results=3)
        print_result(result)


if __name__ == "__main__":
    main()
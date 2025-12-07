from hf_llm import HuggingFaceChatLLM


def main():
    print("Отправляем запрос к модели через LangChain-LLM...")

    llm = HuggingFaceChatLLM()

    resp = llm.invoke("Привет! Объясни в одном предложении, что такое RAG.")
    print("\nОтвет модели:\n")
    print(resp)


if __name__ == "__main__":
    main()

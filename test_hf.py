import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Загружаем токен из .env
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN не найден в .env")

# Создаём клиент
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=token
)

prompt = "Привет! Объясни в 2 предложениях, что такое Retrieval-Augmented Generation (RAG) на русском языке."

print("Отправляем запрос...")

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=80,
    temperature=0.3
)

print("\nОтвет модели:\n")
print(response.choices[0].message["content"])


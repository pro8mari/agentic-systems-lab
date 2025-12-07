import os
from typing import List, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import PrivateAttr


class HuggingFaceChatLLM(LLM):
    """
    Адаптер Hugging Face Chat Completions под интерфейс LangChain LLM.
    """

    # Публичные поля-модели (pydantic)
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_tokens: int = 200
    temperature: float = 0.3

    # Приватный атрибут — НЕ поле pydantic, можно свободно присваивать в __init__
    _client: InferenceClient = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)

        load_dotenv()
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("Переменная HUGGINGFACEHUB_API_TOKEN не найдена в .env")

        self._client = InferenceClient(model=self.model, token=token)

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        """То, что LangChain вызывает внутри llm.invoke()."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message["content"]

import os
import requests
from typing import Optional
from ..base import LLMClientBase


class PortkeyClient(LLMClientBase):
    def __init__(self, model: Optional[str] = None):
        super().__init__(model_name="portkey", model=model or "gpt-4o-mini")
        self.api_key = os.getenv("PORTKEY_API_KEY")
        self.base_url = "https://api.portkey.ai/v1/chat/completions"

        if not self.api_key:
            raise ValueError("Missing PORTKEY_API_KEY in .env")

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            print(f"Warning: model '{self.default_model}' does not support images; fallback to text-only.")

        self.add_message("user", prompt)
        response_text = self._query_text_only()
        self.add_message("assistant", response_text)
        return response_text

    def _query_text_only(self) -> str:
        payload = {
            "model": self.default_model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": False  # streaming 可选
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

import os
import time
import anthropic
from typing import Optional
from dotenv import load_dotenv
from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token

load_dotenv()


class ClaudeClient(LLMClientBase):
    def __init__(self, model: str, api_key: str, max_context_tokens: int = 100000):
        super().__init__(model_name="claude", model=model)
        self.api_key = api_key
        self.max_context_tokens = max_context_tokens

        if not self.api_key:
            raise ValueError("Claude API Key is required for ClaudeClient.")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            print("[Warning] ClaudeClient does not support images yet, ignoring image.")

        self.add_message("user", prompt)
        self.messages = truncate_messages_by_token(self.messages, self.max_context_tokens, self.default_model)

        response_text = self._call_api()
        self.add_message("assistant", response_text)
        return response_text

    def _call_api(self) -> str:
        messages_for_api = [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
            if m["role"] != "system"
        ]

        for attempt in range(3):
            try:
                params = {
                    "model": self.default_model,
                    "max_tokens": self.max_tokens or 1024,
                    "messages": messages_for_api,
                }
                if self.system_prompt:
                    params["system"] = self.system_prompt
                if self.temperature is not None:
                    params["temperature"] = self.temperature
                if self.top_p is not None:
                    params["top_p"] = self.top_p

                response = self.client.messages.create(**params)
                return response.content[0].text.strip()

            except Exception as e:
                print(f"[ClaudeClient] Error: {type(e).__name__} - {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return "Error: Max retries reached"

        return "Error: Max retries reached"

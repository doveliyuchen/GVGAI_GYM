import base64
import os
import time
import requests
from typing import Optional
from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token

class QwenClient(LLMClientBase):
    def __init__(self, model: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(model_name="qwen", model=model or "qwen-plus")
        self.api_key = os.getenv("QWEN_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.chat_endpoint = "/chat/completions"
        self.system_prompt = system_prompt
        self.default_model = model
        if not self.api_key:
            raise ValueError("Missing QWEN_API_KEY in .env")

        if self.system_prompt:
            self.clear_history()
            self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })

    def set_system_prompt(self, system_prompt: str):
        """Reset context and set a new system prompt."""
        self.clear_history()
        self.system_prompt = system_prompt
        self.messages.append({
            "role": "system",
            "content": self.system_prompt
        })

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        self.add_message("user", prompt)

        if image_path:
            response_text = self._query_multimodal(prompt, image_path)
        else:
            response_text = self._query_text_only()

        self.add_message("assistant", response_text)
        return response_text

    def _query_text_only(self) -> str:
        self.messages = truncate_messages_by_token(self.messages, self.max_tokens or 4000, self.default_model)

        payload = {
            "model": self.default_model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url + self.chat_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()

            except requests.exceptions.HTTPError as e:
                if response.status_code in [502, 503, 504] and attempt < 2:
                    print(f"[Warning] Qwen Server error {response.status_code}, retrying after 3 seconds...")
                    time.sleep(3)
                    continue
                else:
                    print(f"[QwenClient] API Error: {e}")
                    return ""
            except Exception as e:
                print(f"[QwenClient] Unexpected Error: {e}")
                return ""

    def _query_multimodal(self, prompt: str, image_path: str) -> str:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]

        payload = {
            "model": self.default_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url + self.chat_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()

            except requests.exceptions.HTTPError as e:
                if response.status_code in [502, 503, 504] and attempt < 2:
                    print(f"[Warning] Qwen Server error {response.status_code}, retrying after 3 seconds...")
                    time.sleep(3)
                    continue
                else:
                    print(f"[QwenClient] API Error: {e}")
                    return ""
            except Exception as e:
                print(f"[QwenClient] Unexpected Error: {e}")
                return ""

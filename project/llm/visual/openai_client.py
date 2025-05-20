import base64
import os
import time
import requests
from typing import Optional
from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token

class OpenAIClient(LLMClientBase):
    def __init__(self, model: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(model_name="openai", model=model or "gpt-4o")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.chat_endpoint = "/chat/completions"
        # self.system_prompt = system_prompt
        self.default_model = model

        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env")

        # if self.system_prompt:
        #     self.clear_history()
        #     self.messages.append({
        #         "role": "system",
        #         "content": self.system_prompt
        #     })

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
        self.messages = truncate_messages_by_token(self.messages, self.max_tokens or 10000, self.default_model)

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

        # print(payload)

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
                    print(f"[Warning] Server error {response.status_code}, retrying after 3 seconds...")
                    time.sleep(3)
                    continue
                else:
                    print(f"[OpenAIClient] API Error: {e}")
                    return ""
            except Exception as e:
                print(f"[OpenAIClient] Unexpected Error: {e}")
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

        # multimodal 只发一轮，不需要truncate
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
                    print(f"[Warning] Server error {response.status_code}, retrying after 3 seconds...")
                    time.sleep(3)
                    continue
                else:
                    print(f"[OpenAIClient] API Error: {e}")
                    return ""
            except Exception as e:
                print(f"[OpenAIClient] Unexpected Error: {e}")
                return ""

if __name__ == "__main__":
    client = OpenAIClient(model="gpt-4o")   # 注意这里的model写的是gpt-4o，跟你client里的default_model一致

    prompt = "You are a professional chess coach. Please explain the basic opening principles in chess in simple words."

    reply = client.query(prompt)

    print("\n=== Model Reply ===")
    print(reply)

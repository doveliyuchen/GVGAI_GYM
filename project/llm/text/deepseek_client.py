import os
import time
import requests
from typing import Optional
from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token

class DeepseekClient(LLMClientBase):
    def __init__(self, model: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(model_name="deepseek", model=model or "deepseek-chat")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"
        self.chat_endpoint = "/chat/completions"
        self.system_prompt = system_prompt
        self.default_model = model or "deepseek-chat"
        
        # Initialize attributes that may be set by client.py
        self.max_tokens = None
        self.temperature = None
        self.top_p = None
        
        if not self.api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY in .env")

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
        if image_path:
            print(f"[Warning] Deepseek '{self.default_model}' does not support images; fallback to text-only.")

        self.add_message("user", prompt)
        response_text = self._query_text_only()
        self.add_message("assistant", response_text)
        return response_text

    def _query_text_only(self) -> str:
        # Use a reasonable context limit for message truncation, not max_tokens (which is for response)
        # DeepSeek models typically have 32k context, so use a conservative limit
        context_limit = 28000  # Leave room for response
        self.messages = truncate_messages_by_token(self.messages, context_limit, self.default_model)

        # Build payload, only include non-None values
        payload = {
            "model": self.default_model,
            "messages": self.messages,
        }
        
        # Only add optional parameters if they are not None
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            payload["top_p"] = self.top_p
            
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
                    print(f"[Warning] Deepseek Server error {response.status_code}, retrying after 3 seconds...")
                    time.sleep(3)
                    continue
                else:
                    # Print more detailed error information for 400 errors
                    error_detail = ""
                    try:
                        error_detail = response.json()
                    except:
                        error_detail = response.text
                    print(f"[DeepseekClient] API Error {response.status_code}: {e}")
                    print(f"[DeepseekClient] Error details: {error_detail}")
                    print(f"[DeepseekClient] Payload sent: {payload}")
                    return ""
            except Exception as e:
                print(f"[DeepseekClient] Unexpected Error: {e}")
                return ""

import os
import sys
import time
from typing import Optional, List, Dict

import httpx
from dotenv import load_dotenv

# Add project root to path for direct execution
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from llm.base import LLMClientBase
from llm.utils.config import truncate_messages_by_token

load_dotenv()


class GeminiClient(LLMClientBase):
    def __init__(self,
                 model: str,
                 api_key: str,
                 system_prompt: Optional[str] = None,
                 max_context_tokens: Optional[int] = 4000,
                 **kwargs):

        super().__init__(model_name="gemini", model=model)

        self.model = model
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

        if not self.api_key:
            raise ValueError("Gemini API Key is required for GeminiClient.")

        self.max_context_tokens = max_context_tokens
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def set_system_prompt(self, system_prompt: str):
        self.clear_history()
        self.system_prompt = system_prompt
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def query(self, prompt: str, image_path: Optional[str] = None, model: Optional[str] = None) -> str:
        if image_path:
            print(f"[Warning] GeminiClient does not support images yet, ignoring image.")

        self.add_message("user", prompt)

        model_to_use = self.model

        self.messages = truncate_messages_by_token(self.messages, self.max_context_tokens, model_to_use)

        response_text = self._query_gemini(model_to_use)
        self.add_message("assistant", response_text)
        return response_text

    def _query_gemini(self, model: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": model,
            "messages": self.messages,
        }

        for attempt in range(3):
            try:
                with httpx.Client() as client:
                    response = client.post(
                        url=self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=300
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"].strip()

            except httpx.HTTPStatusError as e:
                error_message = f"[GeminiClient] API Error: {e.response.status_code} - {e.response.text}"
                print(error_message)
                if attempt < 2:
                    wait_time = 2 ** attempt
                    print(f"Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error: {e.response.status_code}"
            except httpx.RequestError as e:
                error_message = f"[GeminiClient] Request Error: {e}"
                print(error_message)
                if attempt < 2:
                    print(f"Retrying after 5 seconds...")
                    time.sleep(5)
                    continue
                return "Error: Request failed"
            except Exception as e:
                error_message = f"[GeminiClient] Unexpected Error: {type(e).__name__} - {e}"
                print(error_message)
                return "Error: Unexpected issue"

        return "Error: Max retries reached"

    def clear_history(self):
        super().clear_history()
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})


if __name__ == "__main__":
    # The sys.path modification is now at the top of the file
    client = GeminiClient(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

    prompt = "You are a professional chess coach. Please explain the basic opening principles in chess in simple words."

    reply = client.query(prompt)

    print("\n=== Model Reply ===")
    print(reply)

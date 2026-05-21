import base64
import json
import os
import time
import requests
from typing import Optional
from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token


class OpenAICompatClient(LLMClientBase):
    """Base for all OpenAI-compatible HTTP clients (/chat/completions, Bearer auth)."""

    RETRIES = 5
    RETRY_ON = {429, 502, 503, 504}

    def __init__(self, model_name: str, model: str, base_url: str,
                 api_key: str, context_limit: int = 8000):
        super().__init__(model_name=model_name, model=model)
        self.base_url = base_url
        self.api_key = api_key
        self.context_limit = context_limit

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _build_payload(self, messages: list) -> dict:
        payload: dict = {"model": self.default_model, "messages": messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    def _post(self, payload: dict) -> str:
        for attempt in range(self.RETRIES):
            try:
                resp = requests.post(self.base_url, headers=self._headers(),
                                     json=payload, timeout=300)
                resp.raise_for_status()
                response_json = json.loads(resp.content.decode("utf-8", errors="replace"))
                return response_json["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                if resp.status_code in self.RETRY_ON and attempt < self.RETRIES - 1:
                    wait_seconds = 2 ** (attempt + 1)
                    print(f"[{self.model_name}] HTTP {resp.status_code}, retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue
                print(f"[{self.model_name}] HTTP Error {resp.status_code}: {e}")
                return ""
            except Exception as e:
                print(f"[{self.model_name}] Error: {e}")
                return ""
        return ""

    def _vision_payload(self, prompt: str, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        payload: dict = {"model": self.default_model, "messages": [{"role": "user", "content": content}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        self.add_message("user", prompt)
        if image_path:
            payload = self._vision_payload(prompt, image_path)
        else:
            msgs = truncate_messages_by_token(self.messages, self.context_limit, self.default_model)
            payload = self._build_payload(msgs)
        response = self._post(payload)
        self.add_message("assistant", response)
        return response


# ---------------------------------------------------------------------------
# Thin subclasses
# ---------------------------------------------------------------------------

class OpenAIClient(OpenAICompatClient):
    def __init__(self, model: Optional[str] = None, context_limit: int = 10000):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env")
        super().__init__(
            model_name="openai",
            model=model or "gpt-4o",
            base_url="https://api.openai.com/v1/chat/completions",
            api_key=api_key,
            context_limit=context_limit,
        )


class QwenClient(OpenAICompatClient):
    def __init__(self, model: Optional[str] = None):
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("Missing QWEN_API_KEY in .env")
        super().__init__(
            model_name="qwen",
            model=model or "qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            api_key=api_key,
            context_limit=4000,
        )


class DeepseekClient(OpenAICompatClient):
    def __init__(self, model: Optional[str] = None):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY in .env")
        super().__init__(
            model_name="deepseek",
            model=model or "deepseek-chat",
            base_url="https://api.deepseek.com/chat/completions",
            api_key=api_key,
            context_limit=28000,
        )

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            print("[Warning] DeepseekClient does not support images; falling back to text-only.")
        return super().query(prompt, image_path=None)


class GeminiClient(OpenAICompatClient):
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise ValueError("Gemini API Key is required for GeminiClient.")
        super().__init__(
            model_name="gemini",
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            api_key=api_key,
            context_limit=4000,
        )

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            print("[Warning] GeminiClient does not support images yet, ignoring image.")
        return super().query(prompt, image_path=None)



class PortkeyClient(OpenAICompatClient):
    def __init__(self, actual_model_name: str, portkey_api_key: str,
                 virtual_key: str, base_url: str, max_context_tokens: int = 8000):
        if not portkey_api_key:
            raise ValueError("Portkey API Key is required for PortkeyClient.")
        if not virtual_key:
            raise ValueError("Portkey Virtual Key is required for PortkeyClient.")
        super().__init__(
            model_name="portkey",
            model=actual_model_name,
            base_url=base_url.rstrip("/") + "/v1/chat/completions",
            api_key=portkey_api_key,
            context_limit=max_context_tokens,
        )
        self.virtual_key = virtual_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-portkey-api-key": self.api_key,
            "x-portkey-virtual-key": self.virtual_key,
        }

    def _build_payload(self, messages: list) -> dict:
        payload = super()._build_payload(messages)
        if "gemini-pro" in self.default_model:
            payload["thinking"] = {"type": "enabled", "budget_tokens": 32768}
        return payload

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            print("[Warning] PortkeyClient image support not implemented, ignoring image.")
        return super().query(prompt, image_path=None)

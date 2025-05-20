from ..base import LLMClientBase
from ollama import chat, ChatResponse
import base64
import requests
import subprocess
import time
from pathlib import Path
from typing import Optional


class OllamaClient(LLMClientBase):
    def __init__(self, model: Optional[str] = None):
        super().__init__(model_name="ollama", model=model or "gemma3")
        self.base_url = "http://localhost:11434"
        self.ollama_process = None
        self.supports_vision = self._check_vision_support()
        self.ensure_model_available()

    def _check_vision_support(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/api/show",
                params={"name": self.default_model},
                timeout=50
            )
            model_info = response.json()
            return "vision" in model_info.get("parameters", {}).get("capabilities", [])
        except:
            return False

    def ensure_model_available(self):
        try:
            result = subprocess.run(
                ["ollama", "show", self.default_model],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                subprocess.run(["ollama", "pull", self.default_model], check=True)
        except Exception as e:
            raise RuntimeError(f"Error ensuring model availability: {e}")

    def start_ollama_service(self):
        if self.ollama_process is None:
            self.ollama_process = subprocess.Popen(["ollama", "serve"])
            time.sleep(5)

    def shutdown(self):
        if self.ollama_process:
            self.ollama_process.terminate()
            self.ollama_process.wait()
            self.ollama_process = None

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        self.start_ollama_service()
        self.add_message("user", prompt)

        if image_path and self.supports_vision:
            result = self._query_multi_modal(prompt, image_path)
        else:
            result = self._query_text_only(prompt)

        self.add_message("assistant", result)
        return result

    def _query_text_only(self, prompt: str) -> str:
        try:
            response: ChatResponse = chat(
                model=self.default_model,
                messages=self.messages,
                options={"temperature": 0}
            )
            return response.message.content.strip()
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""

    def _query_multi_modal(self, prompt: str, image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            response: ChatResponse = chat(
                model=self.default_model,
                messages=self.messages + [
                    {"role": "user", "content": {"type": "image", "image": image_base64}}
                ],
                options={"temperature": 0}
            )
            return response.message.content.strip()
        except Exception as e:
            print(f"Ollama multimodal API error: {e}")
            return ""

import base64
import os
import time
import requests
from typing import Optional
from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token


class VLLMClient(LLMClientBase):
    """
    vLLM Client with OpenAI-compatible API.
    vLLM is a high-throughput and memory-efficient inference engine for LLMs.
    """
    
    def __init__(
        self, 
        model: Optional[str] = None, 
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_context_tokens: Optional[int] = None
    ):
        super().__init__(model_name="vllm", model=model or "meta-llama/Llama-3.1-8B-Instruct")
        
        # vLLM server configuration
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM often uses "EMPTY" as default
        self.chat_endpoint = "/v1/chat/completions"
        self.default_model = model
        self.max_context_tokens = max_context_tokens or 8000
        
        # Generation parameters (can be overridden)
        self.temperature = 0.0
        self.max_tokens = 2000
        self.top_p = 1.0
        
        if system_prompt:
            self.set_system_prompt(system_prompt)

    def set_system_prompt(self, system_prompt: str):
        """Reset context and set a new system prompt."""
        self.clear_history()
        self.messages.append({
            "role": "system",
            "content": system_prompt
        })

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        Query the vLLM server with a prompt.
        If image_path is provided, will attempt multimodal query (if model supports it).
        """
        self.add_message("user", prompt)

        if image_path:
            response_text = self._query_multimodal(prompt, image_path)
        else:
            response_text = self._query_text_only()

        self.add_message("assistant", response_text)
        return response_text

    def _query_text_only(self) -> str:
        """Text-only query using OpenAI-compatible API."""
        # Truncate messages to fit context window
        truncated_messages = truncate_messages_by_token(
            self.messages, 
            self.max_context_tokens, 
            self.default_model
        )

        payload = {
            "model": self.default_model,
            "messages": truncated_messages,
            "temperature": self.temperature if self.temperature is not None else 0.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2000,
            "top_p": self.top_p if self.top_p is not None else 1.0,
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization header if API key is not "EMPTY"
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Retry logic for robustness
        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url + self.chat_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract response text
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[VLLMClient] Unexpected response format: {result}")
                    return ""

            except requests.exceptions.ConnectionError as e:
                print(f"[VLLMClient] Connection Error: Cannot connect to vLLM server at {self.base_url}")
                print(f"Please ensure vLLM server is running. Error: {e}")
                return ""
                
            except requests.exceptions.HTTPError as e:
                if response.status_code in [502, 503, 504] and attempt < 2:
                    print(f"[Warning] vLLM server error {response.status_code}, retrying after 3 seconds...")
                    time.sleep(3)
                    continue
                else:
                    print(f"[VLLMClient] API Error: {e}")
                    print(f"Response: {response.text if response else 'No response'}")
                    return ""
                    
            except requests.exceptions.Timeout:
                print(f"[VLLMClient] Request timeout after 300 seconds")
                if attempt < 2:
                    print(f"Retrying... (attempt {attempt + 2}/3)")
                    continue
                return ""
                
            except Exception as e:
                print(f"[VLLMClient] Unexpected Error: {e}")
                return ""
        
        return ""

    def _query_multimodal(self, prompt: str, image_path: str) -> str:
        """
        Multimodal query with image support.
        Note: vLLM multimodal support depends on the specific model being served.
        """
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[VLLMClient] Error reading image file: {e}")
            return ""

        # OpenAI-compatible multimodal format
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
            "temperature": self.temperature if self.temperature is not None else 0.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2000,
            "top_p": self.top_p if self.top_p is not None else 1.0,
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(3):
            try:
                response = requests.post(

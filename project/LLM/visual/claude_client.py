import os
import anthropic
from typing import Optional
from dotenv import load_dotenv
from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token
import time

load_dotenv()

class ClaudeClient(LLMClientBase):
    def __init__(self, 
                 model: str,
                 api_key: str,
                 system_prompt: Optional[str] = None,
                 max_context_tokens: Optional[int] = 100000):
        
        super().__init__(model_name="claude", model=model)
        
        self.model = model
        self.api_key = api_key
        self.max_context_tokens = max_context_tokens
        self.max_tokens = None
        self.temperature = None
        self.top_p = None

        if not self.api_key:
            raise ValueError("Claude API Key is required for ClaudeClient.")

        self.client = anthropic.Anthropic(api_key=self.api_key)

        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
        self.system_prompt = system_prompt

    def set_system_prompt(self, system_prompt: str):
        self.clear_history()
        self.system_prompt = system_prompt
        if self.system_prompt:
            self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            print(f"[Warning] ClaudeClient does not support images yet, ignoring image.")

        self.add_message("user", prompt)
        
        self.messages = truncate_messages_by_token(self.messages, self.max_context_tokens, self.model)

        response_text = self._query_text_only()
        self.add_message("assistant", response_text)
        return response_text

    def _query_text_only(self) -> str:
        # Anthropic API uses a different message format
        messages_for_api = []
        for msg in self.messages:
            if msg["role"] == "system":
                # Anthropic doesn't have a system role in the same way as OpenAI.
                # The system prompt is passed as a separate parameter.
                continue
            
            # Ensure only 'role' and 'content' are passed for each message.
            # The Anthropic API is strict and rejects extra keys.
            clean_msg = {"role": msg["role"], "content": msg["content"]}
            messages_for_api.append(clean_msg)


        for attempt in range(3):
            try:
                request_params = {
                    "model": self.model,
                    "max_tokens": self.max_tokens or 1024,
                    "messages": messages_for_api,
                }
                if self.system_prompt:
                    request_params["system"] = self.system_prompt
                if self.temperature is not None:
                    request_params["temperature"] = self.temperature
                if self.top_p is not None:
                    request_params["top_p"] = self.top_p
                
                response = self.client.messages.create(**request_params)
                return response.content[0].text.strip()

            except Exception as e:
                error_message = f"[ClaudeClient] Unexpected Error: {type(e).__name__} - {e}"
                print(error_message)
                if attempt < 2:
                    wait_time = 2 ** attempt
                    print(f"Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return "Error: Unexpected issue"
        
        return "Error: Max retries reached"

    def clear_history(self):
        super().clear_history()
        if self.system_prompt:
            self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })

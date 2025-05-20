import os
import httpx
from typing import Optional, List, Dict
from dotenv import load_dotenv
from ..base import LLMClientBase 
from llm.utils.config import truncate_messages_by_token
# import tiktoken # Not directly used in this class after changes
import time

load_dotenv() # Ensures .env variables are available

class PortkeyClient(LLMClientBase):
    def __init__(self, 
                 actual_model_name: str, # e.g., "gemini-2.0-flash-exp"
                 portkey_api_key: str,    # The general Portkey API key
                 virtual_key: str,        # The specific virtual key for the target model
                 base_url: str,           # Portkey gateway base URL
                 system_prompt: Optional[str] = None,
                 max_context_tokens: Optional[int] = 8000): # Configurable context window for truncation
        
        # model_name for LLMClientBase is 'portkey' to identify the client type,
        # 'model' for LLMClientBase is the actual backend model Portkey will use.
        super().__init__(model_name="portkey", model=actual_model_name) 
        
        self.portkey_api_key = portkey_api_key
        self.virtual_key = virtual_key
        self.base_url = base_url.rstrip('/') + "/v1/chat/completions" # Ensure correct endpoint path
        
        self.actual_model_name_for_payload = actual_model_name # Store the model name for the Portkey payload
        self.max_context_tokens = max_context_tokens # Max tokens for truncation logic

        if not self.portkey_api_key:
            raise ValueError("Portkey API Key is required for PortkeyClient.")
        if not self.virtual_key:
            raise ValueError("Portkey Virtual Key is required for PortkeyClient.")
        if not self.base_url:
            raise ValueError("Portkey Base URL is required for PortkeyClient.")

        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
        self.system_prompt = system_prompt # Store for potential re-use if history is cleared

    def set_system_prompt(self, system_prompt: str):
        self.clear_history() # Clears self.messages
        self.system_prompt = system_prompt
        if self.system_prompt: # Add system prompt back if it exists
            self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            # Gemini models via Portkey might support images, but depends on 'actual_model_name_for_payload'
            # For now, retain warning or adapt if vision is intended for specific Gemini models
            print(f"[Warning] Portkey model '{self.actual_model_name_for_payload}' image support not explicitly handled, ignoring image.")

        self.add_message("user", prompt)
        
        # Truncate messages using the actual model name for tiktoken, if possible, or a default
        self.messages = truncate_messages_by_token(self.messages, self.max_context_tokens, self.actual_model_name_for_payload)

        response_text = self._query_text_only()
        self.add_message("assistant", response_text)
        return response_text

    def _query_text_only(self) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-portkey-api-key": self.portkey_api_key,
            "x-portkey-virtual-key": self.virtual_key,
        }
        
        payload = {
            "model": self.actual_model_name_for_payload, # Use the specific model name for Portkey
            "messages": self.messages,
            # Parameters like temperature, max_tokens (for response), top_p
            # should be added here if they are supported by the Portkey gateway & backend model
            # and if they are intended to be configurable per call.
            # For now, keeping it simple as per "no extra parameters" for Gemini.
            # If self.max_tokens (from LLMClientBase) is meant for response length, add it here:
            # "max_tokens": self.max_tokens, 
        }
        
        # Remove None values from payload, e.g. if max_tokens is not set on client
        # payload = {k: v for k, v in payload.items() if v is not None}


        for attempt in range(3): # Reduced retries for faster feedback during dev
            try:
                # Ensure base_url is just the gateway, and /v1/chat/completions is appended if not already
                # Constructor now handles this.
                response = httpx.post(
                    url=self.base_url, # self.base_url should now be correctly formed
                    headers=headers,
                    json=payload,
                    timeout=300 # Consider making timeout configurable
                )
                response.raise_for_status()
                # Assuming standard OpenAI-like response structure
                return response.json()["choices"][0]["message"]["content"].strip()

            except httpx.HTTPStatusError as e:
                error_message = f"[PortkeyClient] API Error: {e.response.status_code} - {e.response.text}"
                print(error_message)
                if e.response.status_code in [502, 503, 504] and attempt < 2: # Retry on server errors
                    print(f"Retrying after 5 seconds...")
                    time.sleep(5)
                    continue
                else: # Non-retryable HTTP error or max retries reached
                    return f"Error: {e.response.status_code}" 
            except httpx.RequestError as e: # Covers network errors, timeouts not leading to HTTPStatusError
                error_message = f"[PortkeyClient] Request Error: {e}"
                print(error_message)
                if attempt < 2: # Retry on request errors
                    print(f"Retrying after 5 seconds...")
                    time.sleep(5)
                    continue
                return "Error: Request failed"
            except Exception as e: # Catch-all for other unexpected errors (e.g., JSON parsing)
                error_message = f"[PortkeyClient] Unexpected Error: {type(e).__name__} - {e}"
                print(error_message)
                return "Error: Unexpected issue" # Avoid returning empty string on all errors
        
        return "Error: Max retries reached" # Fallback if all retries fail

    def clear_history(self):
        """Clears message history, preserving system prompt if one was set."""
        super().clear_history() # This clears self.messages
        if self.system_prompt: # Add system prompt back if it exists
            self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })

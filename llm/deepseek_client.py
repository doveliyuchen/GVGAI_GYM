import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

if __name__ == "__main__": from base_client import BaseLLMClient
else: from .base_client import BaseLLMClient

class DeepSeekClient(BaseLLMClient):
    """Client for interacting with DeepSeek API using OpenAI SDK"""

    def __init__(self, 
            api_key: Optional[str] = None,  # Optional, will auto-load from .env if not provided
            temperature: float = 0.0, 
            max_tokens: int = 1000, 
            model_id: Optional[str] = 'deepseek-chat'
        ):
        load_dotenv('.env')  # Load env only once here
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY in .env or pass api_key.")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_id = model_id
        self.base_url = "https://api.deepseek.com"
        self.openai_client = None

        super().__init__()

    def _initialize(self) -> None:
        self.openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def query(self, prompt: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a game agent in an environment"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature, 
                max_tokens=self.max_tokens,
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    client = DeepSeekClient()  # No need to manually pass the api_key
    response = client.query("What color best describes the sun? Give a single word answer.")
    print(response)

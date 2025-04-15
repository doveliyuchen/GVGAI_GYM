import os
from typing import Optional
from dotenv import load_dotenv
from portkey_ai import Portkey

if __name__ == "__main__": 
    from base_client import BaseLLMClient
else: 
    from .base_client import BaseLLMClient

class PortkeyClient(BaseLLMClient):
    """Client for interacting with Portkey API using portkey_ai SDK"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        virtual_key: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 1000,
        model_id: Optional[str] = "gpt-4o-mini",
        base_url: Optional[str] = "https://ai-gateway.apps.cloud.rt.nyu.edu"
    ):
        load_dotenv(".env")
        self.api_key = api_key or os.getenv("PORTKEY_API_KEY")
        self.virtual_key = virtual_key or os.getenv("PORTKEY_VIRTUAL_KEY")
        self.base_url = base_url
        if not self.api_key or not self.virtual_key:
            raise ValueError("Portkey API key and virtual key must be provided.")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_id = model_id
        self.portkey_client = None

        super().__init__()

    def _initialize(self) -> None:
        self.portkey_client = Portkey(
            api_key=self.api_key,
            virtual_key=self.virtual_key,
            base_url=self.base_url
        )

    def query(self, prompt: str) -> str:
        try:
            response = self.portkey_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a game agent in an environment"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,  # 注意这里
                stream=False
            )
            return response.choices[0].message["content"]
        except Exception as e:
            print(f"Portkey API error: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    load_dotenv(".env")
    api_key = os.getenv("PORTKEY_API_KEY")
    virtual_key = os.getenv("PORTKEY_VIRTUAL_KEY")
    
    client = PortkeyClient(api_key=api_key, virtual_key=virtual_key)
    response = client.query("What color best describes the ocean? Give a single word answer.")
    print(response)

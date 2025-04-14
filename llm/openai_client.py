import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

if __name__ == "__main__": 
    from base_client import BaseLLMClient
else: 
    from .base_client import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    """Client for interacting with OpenAI's API using OpenAI SDK"""
    
    def __init__(self, 
            api_key: Optional[str] = None, 
            temperature: float = 0, # Generation temperature
            max_tokens: int = 1000, # Maximum tokens in response
            model_id: Optional[str] = "gpt-4o-mini" # Default model
        ):
        load_dotenv('.env')  # Load env only once here
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Openai API key not found. Set OPENAI_API_KEY in .env or pass api_key.")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.model_id = model_id
        self.openai_client = None
        
        super().__init__()

    def _initialize(self) -> None:
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.api_key)

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
            print(f"OpenAI API error: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    load_dotenv('.env')
    api_key = os.getenv("OPENAI_API_KEY")
    
    client = OpenAIClient(api_key)
    response = client.query("What color best describes the ocean? Give a single word answer.")
    print(response)

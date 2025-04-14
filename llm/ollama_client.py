import subprocess
import time
from ollama import chat, ChatResponse

if __name__ == "__main__": from base_client import BaseLLMClient
else: from .base_client import BaseLLMClient

class OllamaClient(BaseLLMClient):
    """Client for interacting with locally running Ollama models"""
    
    def __init__(self, model_id: str = "gemma3", temperature: float = 0.7):
        self.model_id = model_id
        self.temperature = temperature
        self.ollama_process = None
        self.base_url = "http://localhost:11434"
        super().__init__()
    
    def _initialize(self) -> None:
        """Start Ollama service and ensure model is available"""
        self.start_service()
        self.ensure_model_available()
    
    def start_service(self) -> None:
        """Start the Ollama service if not already running"""
        if self.ollama_process is not None: return
            
        try:
            # Check if Ollama is already running
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200: 
                return  # Service is already running
            
        except: pass

        # If exception occurs, start the service    
        print("Starting Ollama service...")
        self.ollama_process = subprocess.Popen(['ollama', 'serve'])
        time.sleep(3)  # Wait for service to start
     
    def ensure_model_available(self) -> None:
        """Ensure the specified model is available locally"""
        try:
            # Check if model exists
            result = subprocess.run(
                ['ollama', 'show', self.model_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8"
            )
            
            # If model doesn't exist, pull it
            if result.returncode != 0:
                print(f"Model '{self.model_id}' not found. Pulling it now...")
                subprocess.run(['ollama', 'pull', self.model_id], check=True)

        except Exception as e:
            raise RuntimeError(f"Failed to check or pull model '{self.model_id}': {str(e)}")
    
    def query(self, prompt: str) -> str:
        try:
            response: ChatResponse = chat(
                model=self.model_id,
                messages=[ { 'role': 'user', 'content': prompt }, ],
                options={ "temperature": self.temperature }
            )

            return response.message.content.strip()
        
        except Exception as e:
            print(f"Ollama query error: {e}")
            return ""
    
    def shutdown(self) -> None:
        """Shut down the Ollama service if it was started by this instance"""
        if self.ollama_process is None: return
            
        print("Shutting down Ollama service...")
        try:
            self.ollama_process.terminate()
            self.ollama_process.wait(timeout=5)
            print("Ollama service has been terminated.")

        except Exception as e:
            print(f"Error shutting down Ollama service: {e}")
            self.ollama_process.kill()  # Force kill if terminate doesn't work
        
        finally:
            self.ollama_process = None

# Example usage
if __name__ == "__main__":
    with OllamaClient(model_id="gemma3:12b") as client:
        response = client.query("What color best describes the sun? Give a single word answer.")
        print(response)
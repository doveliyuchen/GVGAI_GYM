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
        self.started_by_me = False  # Track if we started the ollama service
        
        # Start ollama service first (if not already running)
        self.start_ollama_service()
        
        # Now we can safely check vision support and ensure model is available
        self.supports_vision = self._check_vision_support()
        self.ensure_model_available()

    def _is_ollama_running(self) -> bool:
        """Check if ollama service is already running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _check_vision_support(self) -> bool:
        """Check if the model supports vision capabilities."""
        try:
            response = requests.get(
                f"{self.base_url}/api/show",
                params={"name": self.default_model},
                timeout=50
            )
            model_info = response.json()
            return "vision" in model_info.get("parameters", {}).get("capabilities", [])
        except Exception as e:
            print(f"Warning: Could not check vision support: {e}")
            return False

    def ensure_model_available(self):
        """Ensure the model is available locally, pull it if not."""
        try:
            print(f"Checking if model '{self.default_model}' is available...")
            
            # Use 'ollama list' to check available models
            result = subprocess.run(
                ["ollama", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            # Check if model is in the list
            if self.default_model not in result.stdout:
                print(f"Model '{self.default_model}' not found locally. Attempting to pull...")
                
                # Try to pull the model
                pull_result = subprocess.run(
                    ["ollama", "pull", self.default_model],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600  # 10 minutes timeout for pulling
                )
                
                if pull_result.returncode == 0:
                    print(f"✓ Successfully pulled model '{self.default_model}'")
                else:
                    # Model pull failed - provide helpful error message
                    error_msg = pull_result.stderr.strip()
                    print(f"✗ Failed to pull model '{self.default_model}'")
                    print(f"Error: {error_msg}")
                    
                    # Suggest available models
                    print("\nTrying to list available models...")
                    list_result = subprocess.run(
                        ["ollama", "list"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=10
                    )
                    if list_result.returncode == 0:
                        print("Currently installed models:")
                        print(list_result.stdout)
                    
                    raise RuntimeError(
                        f"Model '{self.default_model}' not found and failed to pull. "
                        f"Please check the model name or install it manually with: ollama pull {self.default_model}"
                    )
            else:
                print(f"✓ Model '{self.default_model}' is already available")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Timeout while checking/pulling model '{self.default_model}'")
        except Exception as e:
            raise RuntimeError(f"Error ensuring model availability: {e}")

    def start_ollama_service(self):
        """Start ollama service if it's not already running."""
        # First check if ollama is already running
        if self._is_ollama_running():
            print("✓ Ollama service is already running (started by another process)")
            self.started_by_me = False  # We didn't start it, so we won't shut it down
            return
        
        # Service not running, try to start it
        if self.ollama_process is None:
            print("Starting Ollama service...")
            try:
                # Start the process without capturing output to avoid blocking
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for service to be ready
                max_retries = 15  # Increased timeout
                for i in range(max_retries):
                    time.sleep(1)
                    
                    # Check if process is still running
                    if self.ollama_process.poll() is not None:
                        # Process terminated - get the error
                        stderr_output = self.ollama_process.stderr.read() if self.ollama_process.stderr else ""
                        print(f"✗ Ollama process terminated unexpectedly")
                        if stderr_output:
                            print(f"Error output:\n{stderr_output}")
                        raise RuntimeError(
                            f"Ollama serve process terminated. Error: {stderr_output}\n"
                            "This might be caused by:\n"
                            "1. Port 11434 already in use (another ollama instance running)\n"
                            "2. Permission issues\n"
                            "3. Ollama installation problem\n"
                            "Try: killall ollama; ollama serve"
                        )
                    
                    if self._is_ollama_running():
                        print("✓ Ollama service started successfully by this client")
                        self.started_by_me = True  # Mark that we started it
                        return
                    print(f"Waiting for Ollama service to start... ({i+1}/{max_retries})")
                
                # If we get here, service didn't start - try to get error info
                if self.ollama_process and self.ollama_process.stderr:
                    stderr_output = self.ollama_process.stderr.read()
                    if stderr_output:
                        print(f"Ollama stderr output:\n{stderr_output}")
                
                raise RuntimeError(
                    "Ollama service failed to start within timeout (15 seconds).\n"
                    "Possible causes:\n"
                    "1. Port 11434 already in use - try: lsof -i :11434 or netstat -tulpn | grep 11434\n"
                    "2. Another ollama instance running - try: killall ollama\n"
                    "3. Try starting manually: ollama serve\n"
                    "4. Check logs with: journalctl -u ollama (if running as service)"
                )
                
            except FileNotFoundError:
                raise RuntimeError(
                    "Ollama command not found. Please ensure Ollama is installed and in your PATH. "
                    "Visit https://ollama.com for installation instructions."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to start Ollama service: {e}")

    def shutdown(self):
        """
        Shutdown ollama service ONLY if it was started by this client instance.
        This ensures we don't interfere with other experiments running in parallel.
        """
        if self.ollama_process and self.started_by_me:
            print("Shutting down Ollama service (started by this client)...")
            self.ollama_process.terminate()
            self.ollama_process.wait()
            self.ollama_process = None
            self.started_by_me = False
        elif self.ollama_process:
            print("Not shutting down Ollama service (was already running before this client)")
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

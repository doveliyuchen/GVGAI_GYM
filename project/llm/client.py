import os
import base64
import requests
import subprocess
import time
from typing import Dict, Optional
from pathlib import Path
from openai import OpenAI
from ollama import chat, ChatResponse
import sys


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 
sys.path.insert(0, str(project_root))
from utils.config import load_environment

class LLMClient:
    def __init__(self, model_name: str, model: Optional[str] = None):
        load_environment() 
        self.model_configs = {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-4o-mini",
                "chat_endpoint": "/chat/completions"
            },
            "deepseek": {
                "env_var": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",
                "default_model": "deepseek-chat",
                "chat_endpoint": "/chat/completions"
            },
            "qwen": {
                "env_var": "QWEN_API_KEY",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "default_model": "qwen-plus",
                "chat_endpoint": "/chat/completions"
            },
            "claude": {
                "env_var": "CLAUDE_API_KEY",
                "base_url": "https://api.anthropic.com/v1",
                "default_model": "claude-3-opus-20240229",
                "chat_endpoint": "/chat/completions"
            },
            "groq": {
                "env_var": "GROQ_API_KEY",
                "base_url": "https://api.groq.com/openai/v1",
                "default_model": "llama-3.3-70b-versatile",
                "chat_endpoint": "/chat/completions"
            },
            "ollama": {
                "env_var": None,
                "base_url": "http://localhost:11434",
                "default_model": "gemma3",
                "chat_endpoint": "/api/generate",
                "dynamic_vision_check": True,
            }
        }
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}. Available options: {list(self.model_configs.keys())}")
        
        self.model_name = model_name
        self.config = self.model_configs[model_name]
        self.default_model = model or self.config["default_model"]
        self.supports_vision = self._check_vision_support()
                # Handle API key requirements
        if self.config["env_var"]:
            self.api_key = os.getenv(self.config["env_var"])
            if not self.api_key:
                raise ValueError(f"API key not found for {self.config['env_var']}, please check your .env file")
        else:
            self.api_key = None

        self.ollama_process = None 


    def _check_vision_support(self) -> bool:
        """动态检测模型是否支持视觉功能"""
        if self.config.get("dynamic_vision_check", False):
            try:
                # 获取模型详细信息
                response = requests.get(
                    f"{self.config['base_url']}/api/show",
                    params={"name": self.default_model},
                    timeout=50
                )
                model_info = response.json()
                
                # 检查是否包含视觉相关参数
                return "vision" in model_info.get("parameters", {}).get("capabilities", [])
            except:
                return False
        return self.config.get("vision_support", False)
        
    def ensure_model_available(self):
        """
        Ensure the specified model is available locally.
        If not, pull the model using the Ollama CLI.
        """
        if self.model_name != "ollama":
            return  # 只对 Ollama 模型执行此操作

        # print(f"Checking if model '{self.default_model}' is available...")
        try:
            # 调用 Ollama CLI 检查模型是否存在
            result = subprocess.run(
                ['ollama', 'show', self.default_model],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                # print(f"Model '{self.default_model}' is already available.")
                return
        except Exception as e:
            print(f"Error checking model availability: {e}")

        # 如果模型不存在，尝试下载
        # print(f"Model '{self.default_model}' not found. Pulling it now...")
        try:
            subprocess.run(['ollama', 'pull', self.default_model], check=True)
            # print(f"Model '{self.default_model}' has been successfully pulled.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to pull model '{self.default_model}': {e.stderr}")

    def start_ollama_service(self):
        """启动 Ollama 服务"""
        if self.model_name != "ollama":
            print("Ollama service is only applicable for the 'ollama' model.")
            return
        
        if self.ollama_process is not None:
            # print("Ollama service is already running.")
            return
        
        # print("Starting Ollama service...")
        self.ollama_process = subprocess.Popen(['ollama', 'serve'])
        time.sleep(5)  # 等待服务启动
        # print("Ollama service has been started.")

    def shutdown_ollama_service(self):
        """关闭 Ollama 服务"""
        if self.model_name != "ollama":
            print("Ollama service is only applicable for the 'ollama' model.")
            return
        
        if self.ollama_process is None:
            print("Ollama service is not running.")
            return
        
        print("Shutting down Ollama service...")
        try:
            # 终止子进程
            self.ollama_process.terminate()
            self.ollama_process.wait()
            print("Ollama service has been terminated.")
        except Exception as e:
            print(f"Error shutting down Ollama service: {e}")
        finally:
            self.ollama_process = None


    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path and not self.supports_vision:
            raise ValueError(f"{self.default_model} does not support multi-modal inputs")
        
        try:
            if self.model_name == "ollama":
                if self.ollama_process is None:
                    self.start_ollama_service()  # 自动启动 Ollama 服务
                    self.ensure_model_available()
                
                self.ensure_model_available()
                return self._query_ollama(prompt, image_path)
            else:
                
                return self._query_text_only(prompt)
        except Exception as e:
            print(f"API request failed: {str(e)}")
            return ""
        # finally:
        #     if self.model_name == "ollama":
        #         self.shutdown_ollama_service()  # 自动关闭 Ollama 服务

    def _query_ollama(self, prompt: str,image_path: Optional[str] = None) -> str:
        """
        Handle queries using the Ollama package.
        
        :param prompt: The text input for the model.
        :return: The response from the Ollama model as a string.
        """
        try:
            response: ChatResponse = chat(model=self.default_model, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            return response.message.content.strip()
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return ""

    def _query_text_only(self, prompt: str) -> str:
        if self.model_name == "ollama":
            return self._query_ollama(prompt)
        
        payload = {
            "model": self.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1000,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            self.config["base_url"] + self.config["chat_endpoint"],
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


    def _query_multi_modal(self, prompt: str, image_path: str) -> str:

        with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            
        if self.config["model_name"] == "ollama":
            
            payload = {
                "model": self.default_model,
                "prompt": prompt,
                "images": [base64_image],  # Ollama使用images字段
                "stream": False
            }
            response = requests.post(
                self.config["base_url"] + "/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()["response"]
        
       
        
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
        
        payload = {
            "model": self.default_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 1000,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            self.config["base_url"] + self.config["chat_endpoint"],
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    # 测试所有模型
    llm_ls = [ "ollama"]



    for llm in llm_ls:

        print(f"\nTesting {llm.upper()} model:")
        client = LLMClient(model_name=llm)
        # client.shutdown_ollama_service()
        
        # 测试文本查询
        text_result = client.query("Compare 9.11 and 9.8, which is larger?")
        print("Text Response:", text_result)
            


import os
import base64
import requests
from typing import Dict, Optional
from pathlib import Path
from openai import OpenAI  # Import the new OpenAI client
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # Assuming project structure: your_project/llm/client.py
sys.path.insert(0, str(project_root))
from utils.config import load_environment


class LLMClient:
    def __init__(self, model_name: str):
        load_environment()  # Ensure environment variables are loaded
        """
        Initialize LLM client
        :param model_name: Supported model names (openai/deepseek/qwen/claude)
        """
        self.model_configs = {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-4o",  # Updated for Vision API
                "headers": lambda key: {"Authorization": f"Bearer {key}"},
                "payload": {
                    "temperature": 0,
                    "max_tokens": 1000
                }
            },
            "deepseek": {
                "env_var": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",  # 根路径
                "default_model": "deepseek-reasoner",  # 使用官方示例中的模型名称
                "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                "payload": {
                    "temperature": 0,
                    "max_tokens": 2000,
                    "top_p": 1.0
                }
            },
            "qwen": {
                "env_var": "QWEN_API_KEY",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 兼容模式路径
                "default_model": "qwen-omni-turbo",  # 使用官方示例中的模型名称
                "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                "payload": {
                    "temperature": 0,
                    "max_tokens": 1000
                }
            },
            "claude": {
                "env_var": "CLAUDE_API_KEY",
                "base_url": "https://api.anthropic.com/v1/complete",
                "default_model": "claude-3-opus",
                "headers": lambda key: {"x-api-key": key},
                "payload": {
                    "temperature": 0,
                    "max_tokens_to_sample": 1000
                }
            }
        }
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}. Available options: {list(self.model_configs.keys())}")
        self.model_name = model_name
        self.config = self.model_configs[model_name]
        self.api_key = os.getenv(self.config["env_var"])
        if not self.api_key:
            raise ValueError(f"API key not found for {self.config['env_var']}, please check your .env file")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.config["base_url"]
        )

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Unified query interface"""
        try:
            if self.model_name == "deepseek":
                return self._query_deepseek(prompt)
            elif self.model_name in ["openai", "qwen"]:
                return self._query_multi_modal(prompt, image_path)
            else:
                return self._query_text_only(prompt)
        except Exception as e:
            print(f"API request failed: {str(e)}")
            return ""

    def _query_deepseek(self, prompt: str) -> str:
        """Handle DeepSeek queries (streaming with reasoning content)"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.config["default_model"],
                messages=messages,
                stream=True  # 启用流式输出
            )
            reasoning_content = ""
            content = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                elif hasattr(delta, "content") and delta.content:
                    content += delta.content
            # print("Reasoning Content:", reasoning_content)  # 打印推理内容（可选）
            return content.strip()
        except Exception as e:
            print(f"DeepSeek API 错误: {e}")
            return ""

    def _query_multi_modal(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Handle multi-modal queries (text + image)"""
        try:
            messages = []
            content = []

            # 添加文本内容
            if prompt:
                content.append({"type": "text", "text": prompt})

            # 添加图像内容（如果提供了图像路径）
            if image_path:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            # 构建消息结构
            messages.append({"role": "user", "content": content})

            # 调用 API
            if self.model_name == "openai":
                response = self.client.chat.completions.create(
                    model=self.config["default_model"],
                    messages=messages,
                    max_tokens=self.config["max_token"]
                )
                return response.choices[0].message.content.strip()

            elif self.model_name == "qwen":
                completion = self.client.chat.completions.create(
                    model=self.config["default_model"],
                    messages=messages,
                    stream=True,  # 启用流式输出
                    max_tokens=self.config["max_token"]

                )
                result = ""
                for chunk in completion:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:  # 确保 content 非空
                            result += delta.content
                return result.strip()

        except Exception as e:
            print(f"{self.model_name.capitalize()} API 错误: {e}")
            return ""

    def _query_text_only(self, prompt: str) -> str:
        """Handle text-only queries"""
        try:
            payload = self._build_payload(prompt)
            response = requests.post(
                self.config["base_url"],
                headers=self.config["headers"](self.api_key),
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            print("Response Data:", response.text if 'response' in locals() else "No response received.")
            return ""
        except KeyError as e:
            print(f"Response parsing failed: {str(e)}")
            return ""

    def _build_payload(self, prompt: str, image_path: Optional[str] = None) -> Dict:
        """Build model-specific payload"""
        payload = self.config["payload"].copy()
        if self.model_name == "deepseek":
            payload["model"] = self.config["default_model"]
            payload["messages"] = [{"role": "user", "content": prompt}]  # 使用 messages 字段
        elif self.model_name == "openai":
            payload["model"] = self.config["default_model"]
            messages = [{"role": "user", "content": prompt}]
            if image_path:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            payload["messages"] = messages
        elif self.model_name in ["qwen", "claude"]:
            payload["model"] = self.config["default_model"]
            payload["prompt"] = prompt
        return payload

    def _parse_response(self, response: Dict) -> str:
        """Parse response from different models"""
        if self.model_name == "deepseek":
            try:
                return response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except (KeyError, IndexError):
                print("Response parsing failed: Unexpected response format.")
                return ""
        elif self.model_name == "openai":
            return response["choices"][0]["message"]["content"].strip()
        elif self.model_name in ["qwen", "claude"]:
            return response.get("output", {}).get("text", "").strip()


if __name__ == "__main__":
    # 初始化客户端
    client = LLMClient(model_name="deepseek")  # 或者 "openai" / "qwen"

    # 测试纯文本查询
    text_prompt = "9.11 和 9.8，哪个更大？"
    text_result = client.query(text_prompt)
    print("Text Response:", text_result)

    # 测试多模态查询
    image_path = "../game_frames/VisionAgent_step0.png"  # 替换为你的图片路径
    image_prompt = "这张图片的内容是什么？"
    image_result = client.query(image_prompt, image_path=image_path)
    print("Image Response:", image_result)

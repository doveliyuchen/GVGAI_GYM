import requests
from typing import Dict, Optional

import sys
import os
from pathlib import Path

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()

# 计算项目根目录（假设项目结构为 your_project/llm/client.py）
project_root = current_file.parent.parent  # 向上两级到 your_project

# 将项目根目录添加到Python路径
sys.path.insert(0, str(project_root))

# 现在可以使用绝对导入
from utils.config import load_environment

class LLMClient:
    def __init__(self, model_name: str):
        load_environment()  # 确保环境变量已加载
        """
        初始化LLM客户端
        :param model_name: 支持的模型名称 (openai/deepseek/qwen/claude)
        """
        self.model_configs = {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1/chat/completions",
                "default_model": "gpt-4o-mini",
                "headers": lambda key: {"Authorization": f"Bearer {key}"},
                "payload": {
                    "temperature": 0,
                    "max_tokens": 1000
                }
            },
            "deepseek": {
                "env_var": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com/v1/chat/completions",
                "default_model": "deepseek-chat",
                "headers": lambda key: {"Authorization": f"Bearer {key}"},
                "payload": {
                    "temperature": 0,
                    "max_tokens": 2000,
                    "top_p": 1.0
                }
            },
            "qwen": {
                "env_var": "QWEN_API_KEY",
                "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                "default_model": "qwen-max",
                "headers": lambda key: {"Authorization": f"Bearer {key}"},
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
            raise ValueError(f"不支持的模型: {model_name}。可用选项: {list(self.model_configs.keys())}")

        self.model_name = model_name
        self.config = self.model_configs[model_name]
        self.api_key = os.getenv(self.config["env_var"])

        if not self.api_key:
            raise ValueError(f"未找到环境变量 {self.config['env_var']}，请检查.env文件配置")

    def query(self, prompt: str) -> str:
        """统一查询接口"""
        try:
            response = requests.post(
                self.config["base_url"],
                headers=self.config["headers"](self.api_key),
                json=self._build_payload(prompt),
                timeout=30
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {str(e)}")
            return ""
        except KeyError as e:
            print(f"响应解析失败: {str(e)}")
            return ""

    def _build_payload(self, prompt: str) -> Dict:
        """构建模型专用请求体"""
        payload = self.config["payload"].copy()

        if self.model_name in ["openai", "deepseek"]:
            payload["model"] = self.config["default_model"]
            payload["messages"] = [{"role": "user", "content": prompt}]
        elif self.model_name == "qwen":
            payload["model"] = self.config["default_model"]
            payload["input"] = {"prompt": prompt}
        elif self.model_name == "claude":
            payload["model"] = self.config["default_model"]
            payload["prompt"] = prompt

        return payload

    def _parse_response(self, response: Dict) -> str:
        """解析不同模型的响应"""
        if self.model_name in ["openai", "deepseek"]:
            return response["choices"][0]["message"]["content"].strip()
        elif self.model_name == "qwen":
            return response["output"]["text"].strip()
        elif self.model_name == "claude":
            return response["completion"].strip()


if __name__ == "__main__":
    # 使用示例
    client = LLMClient(model_name="deepseek")
    result = client.query("请用Python写一个快速排序函数")
    print(result)

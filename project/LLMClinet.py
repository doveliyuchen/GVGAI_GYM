import os
import requests

class LLMClient:
    def __init__(self, model_name="default"):
        self.model_name = model_name
        self.api_key = self._get_api_key()

    def _get_api_key(self):
        """从环境变量中获取 API 密钥"""
        if self.model_name == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.model_name == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY")
        elif self.model_name == "qwen":
            return os.getenv("QWEN_API_KEY")
        elif self.model_name == "claude":
            return os.getenv("CLAUDE_API_KEY")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def query(self, prompt):
        """根据模型名称调用对应的 API"""
        if self.model_name == "openai":
            return self._query_openai(prompt)
        elif self.model_name == "deepseek":
            return self._query_deepseek(prompt)
        elif self.model_name == "qwen":
            return self._query_qwen(prompt)
        elif self.model_name == "claude":
            return self._query_claude(prompt)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _query_openai(self, prompt):
        """调用 OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4o-mini",  # 替换为实际使用的 OpenAI 模型
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 10,
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]

    def _query_deepseek(self, prompt):
        """调用 DeepSeek API"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "deepseek-reasoner",
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0
        }
        response = requests.post(url, headers=headers, json=data)
        # print(response.json())
        return response.json()["choices"][0]["message"]["content"]

    def _query_qwen(self, prompt):
        """调用 Qwen API"""
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "qwen-max",  # 替换为实际的 Qwen 模型名称
            "input": {"prompt": prompt},
            "parameters": {"temperature": 0, "max_tokens": 100},
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()["output"]["text"]

    def _query_claude(self, prompt):
        """调用 Claude API"""
        url = "https://api.anthropic.com/v1/complete"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        data = {
            "model": "claude-v1",  # 替换为实际的 Claude 模型名称
            "prompt": prompt,
            "temperature": 0,
            "max_tokens_to_sample": 10,
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()["completion"]

# 测试
if __name__ == "__main__":

    llm_client = LLMClient(model_name="deepseek")  # 支持 "openai", "deepseek", "qwen", "claude"
    response = llm_client.query("你好，请生成一个简单的 Python 函数，直接生成")
    print(response)

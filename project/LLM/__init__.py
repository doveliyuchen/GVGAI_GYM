"""
llm 包初始化文件

导出接口：
- LLMClient 类
- get_available_models() 函数
"""

# 显式导入关键组件（Python 3.7+ 推荐方式）
from .client import LLMClient

# 定义公开接口（可选但推荐）
__all__ = [
    "LLMClient",
    "get_available_models"
]

# 包版本信息
__version__ = "1.0.0"

# 包初始化逻辑
print(f"Initializing llm package v{__version__}")

def get_available_models():
    """获取当前支持的模型列表"""
    return ["openai", "deepseek", "qwen", "claude"]

from dotenv import load_dotenv
import os


def load_environment():
    """加载环境变量配置"""
    env_loaded = load_dotenv()  # 自动搜索项目根目录的.env

    if not env_loaded:
        raise EnvironmentError("⚠️ 未找到.env文件，请在项目根目录创建")

    # 验证必要环境变量
    # required_vars = [
    #     "OPENAI_API_KEY",
    #     "DEEPSEEK_API_KEY",
    #     "QWEN_API_KEY",
    #     "CLAUDE_API_KEY",
    #     ""
    # ]

    # missing = [var for var in required_vars if not os.getenv(var)]
    # if missing:
    #     raise EnvironmentError(f"缺少必要环境变量: {missing}")

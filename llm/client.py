import os
from pathlib import Path
from dotenv import load_dotenv
from llm.utils.config import get_profile_config

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from llm.visual.claude_client import ClaudeClient
from llm.visual.openai_compat import (
    OpenAIClient, QwenClient, DeepseekClient,
    GeminiClient, PortkeyClient,
)
from llm.visual.vllm_client import VLLMClient


def create_client_from_config(profile: str):
    """Create a configured LLM client from a profile name in llm_config.json."""
    config = get_profile_config(profile)
    params = config.get("parameters", {})

    client_type = config.get("client_type", profile)
    if profile.startswith("portkey-") or client_type == "portkey":
        client_type = "portkey"

    if client_type == "ollama":
        from llm.visual.ollama_client import OllamaClient
        client = OllamaClient(model=config["model"])

    elif client_type == "openai":
        client = OpenAIClient(
            model=config["model"],
            context_limit=params.get("max_context_tokens", 10000),
        )

    elif client_type == "qwen":
        client = QwenClient(model=config["model"])

    elif client_type == "deepseek":
        client = DeepseekClient(model=config["model"])

    elif client_type == "gemini":
        api_key_name = config.get("gemini_api_key")
        if not api_key_name:
            raise ValueError(f"Profile '{profile}' is missing 'gemini_api_key' in llm_config.json")
        api_key = os.getenv(api_key_name)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_name}' for Gemini API key not found.")
        client = GeminiClient(model=config["model"], api_key=api_key)

    elif client_type == "claude":
        api_key_name = config.get("claude_api_key_env_var", "ANTHROPIC_API_KEY")
        api_key = os.getenv(api_key_name)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_name}' for Claude API key not found.")
        client = ClaudeClient(
            model=config["model"],
            api_key=api_key,
            max_context_tokens=params.get("max_context_tokens", 100000),
        )

    elif client_type == "vllm":
        model_val = config.get("model")
        # Allow env-var indirection: if the value is an env var name (all caps + underscores), resolve it
        if model_val and model_val == model_val.upper() and not model_val.startswith("/"):
            model_val = os.getenv(model_val, model_val)
        base_url_val = config.get("base_url") or os.getenv("VLLM_BASE_URL")
        enable_thinking = params.get("enable_thinking")  # None if not set
        client = VLLMClient(
            model=model_val,
            base_url=base_url_val,
            api_key=config.get("api_key"),
            model_name=profile,
            max_model_len=config.get("max_model_len"),
            gpu_memory_utilization=config.get("gpu_memory_utilization"),
            enable_thinking=enable_thinking,
            thinking_budget=params.get("thinking_budget"),
        )

    elif client_type == "portkey":
        portkey_base_url = config.get("portkey_base_url")
        api_key_env_var = config.get("portkey_api_key_env_var")
        virtual_key_env_var = config.get("virtual_key_env_var")
        actual_model_name = config.get("actual_model_name")

        if not all([portkey_base_url, api_key_env_var, virtual_key_env_var, actual_model_name]):
            raise ValueError(
                f"Profile '{profile}' (portkey) is missing one or more required fields: "
                "'portkey_base_url', 'portkey_api_key_env_var', 'virtual_key_env_var', 'actual_model_name'."
            )

        # Narrow Optional values for type checkers after runtime validation above
        portkey_base_url = str(portkey_base_url)
        api_key_env_var = str(api_key_env_var)
        virtual_key_env_var = str(virtual_key_env_var)
        actual_model_name = str(actual_model_name)

        portkey_api_key = os.getenv(api_key_env_var)
        virtual_key = os.getenv(virtual_key_env_var)
        if not portkey_api_key:
            raise ValueError(f"Environment variable '{api_key_env_var}' not found (profile '{profile}').")
        if not virtual_key:
            raise ValueError(f"Environment variable '{virtual_key_env_var}' not found (profile '{profile}').")

        client = PortkeyClient(
            actual_model_name=actual_model_name,
            portkey_api_key=portkey_api_key,
            virtual_key=virtual_key,
            base_url=portkey_base_url,
            max_context_tokens=params.get("max_context_tokens", 8000),
        )

    else:
        raise ValueError(f"Unsupported client_type '{client_type}' for profile '{profile}'.")

    # Apply generation parameters from config (only override if explicitly set)
    if params.get("temperature") is not None:
        client.temperature = params["temperature"]
    if params.get("max_tokens") is not None:
        client.max_tokens = params["max_tokens"]
    if params.get("top_p") is not None:
        client.top_p = params["top_p"]
    if params.get("system_prompt"):
        client.set_system_prompt(params["system_prompt"])

    return client

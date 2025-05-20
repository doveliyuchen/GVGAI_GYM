import json
import os
from pathlib import Path
import tiktoken
from typing import List,Dict
from dotenv import load_dotenv

load_dotenv()


def load_llm_config(path: str = None) -> dict:
    """
    Load LLM configuration for multiple APIs from a JSON file.
    If no path is provided, read from .env `LLM_CONFIG_PATH`, else default to 'project/llm_config.json'.
    Returns a dict of all available profiles.
    """
    # Default path changed to 'project/llm_config.json' to be relative to GVGAI_LLM (CWD)
    config_path = path or os.getenv("LLM_CONFIG_PATH", "project/llm_config.json")
    full_path = Path(config_path) # Path will resolve this relative to CWD

    if not full_path.exists():
        raise FileNotFoundError(f"LLM config file not found at: {full_path.resolve()}")

    with open(full_path, "r") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid LLM config: expected a JSON object with model profiles.")

    return config


def get_profile_config(profile: str, path: str = None) -> dict:
    """
    Return configuration for a specific profile key (e.g., 'openai', 'ollama')
    """
    all_profiles = load_llm_config(path)
    if profile not in all_profiles:
        raise KeyError(f"LLM config profile '{profile}' not found.")
    return all_profiles[profile]

def truncate_messages_by_token(messages: List[Dict[str, str]], max_tokens: int, model: str) -> List[Dict[str, str]]:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    system_prompt = None
    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]
        messages = messages[1:]  # exclude system for now

    truncated = []
    total = 0

    for msg in reversed(messages):
        token_count = len(enc.encode(msg["content"])) + 3
        if total + token_count > max_tokens:
            break
        truncated.insert(0, msg)
        total += token_count

    if system_prompt:
        truncated.insert(0, system_prompt)

    return truncated

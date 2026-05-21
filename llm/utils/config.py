import json
import os
from pathlib import Path
import tiktoken
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def load_llm_config(path: str = None) -> dict:
    """Load all LLM profiles from llm_config.json."""
    if path:
        full_path = Path(path)
    else:
        env_path = os.getenv("LLM_CONFIG_PATH")
        if env_path:
            full_path = Path(env_path)
        else:
            current_file_dir = Path(__file__).parent  # <repo_root>/llm/utils/
            full_path = current_file_dir.parent.parent / "llm_config.json"

    if not full_path.exists():
        raise FileNotFoundError(f"LLM config file not found at: {full_path.resolve()}")

    with open(full_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid LLM config: expected a JSON object with model profiles.")

    return config


def get_profile_config(profile: str, path: str = None) -> dict:
    """Return config for a specific profile, with model name resolved."""
    all_profiles = load_llm_config(path)
    if profile not in all_profiles:
        raise KeyError(f"LLM config profile '{profile}' not found.")

    profile_config = all_profiles[profile].copy()

    model_name = profile_config.get("actual_model_name", profile_config.get("model"))
    # vllm and ollama need the full model identifier (e.g. "meta-llama/Llama-3.1-8B-Instruct")
    # Only strip the org prefix for portkey-style profiles where it's a routing slug
    if model_name and '/' in model_name and profile_config.get("client_type") not in ("vllm", "ollama"):
        model_name = model_name.split('/')[-1]
    if model_name:
        profile_config['model'] = model_name

    return profile_config


def truncate_messages_by_token(messages: List[Dict[str, str]], max_tokens: int, model: str) -> List[Dict[str, str]]:
    if not messages:
        return messages

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    system_msg = None
    rest = messages
    if messages and messages[0]["role"] == "system":
        system_msg = messages[0]
        rest = messages[1:]

    system_tokens = len(enc.encode(system_msg["content"])) + 3 if system_msg else 0
    total = system_tokens
    truncated = []

    for msg in reversed(rest):
        token_count = len(enc.encode(msg["content"])) + 3
        if total + token_count > max_tokens:
            break
        truncated.insert(0, msg)
        total += token_count

    if not truncated and rest:
        print("[Warning] Message truncation: keeping only the most recent message.")
        truncated = [rest[-1]]

    if system_msg:
        truncated.insert(0, system_msg)

    if not truncated:
        print("[Error] All messages were truncated.")
        return [{"role": "user", "content": "Hello"}]

    return truncated

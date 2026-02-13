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
    If no path is provided, read from .env `LLM_CONFIG_PATH`, else default to 'llm_config.json' in project directory.
    Returns a dict of all available profiles.
    """
    if path:
        # Use provided path as-is
        full_path = Path(path)
    else:
        # Check environment variable first
        env_path = os.getenv("LLM_CONFIG_PATH")
        if env_path:
            full_path = Path(env_path)
        else:
            # Default: find llm_config.json relative to this file's location
            # This file is in project/llm/utils/config.py, so go up 2 levels to project/
            current_file_dir = Path(__file__).parent  # project/llm/utils/
            project_dir = current_file_dir.parent.parent  # project/
            full_path = project_dir / "llm_config.json"

    if not full_path.exists():
        raise FileNotFoundError(f"LLM config file not found at: {full_path.resolve()}")

    with open(full_path, "r") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid LLM config: expected a JSON object with model profiles.")

    return config


def get_profile_config(profile: str, path: str = None) -> dict:
    """
    Return configuration for a specific profile, resolving the model name.

    For Portkey profiles with model names like "@scope/model-name", this function
    extracts just "model-name". It prefers "actual_model_name" if present,
    otherwise it uses "model".
    """
    all_profiles = load_llm_config(path)
    if profile not in all_profiles:
        raise KeyError(f"LLM config profile '{profile}' not found.")
    
    profile_config = all_profiles[profile].copy()

    # Resolve the model name, preferring 'actual_model_name'
    model_name = profile_config.get("actual_model_name", profile_config.get("model"))
    
    if model_name and '/' in model_name:
        # Extract the name part from a scoped model name (e.g., "@scope/name" -> "name")
        model_name = model_name.split('/')[-1]
    
    # Update the 'model' key with the resolved name for downstream use
    if model_name:
        profile_config['model'] = model_name

    return profile_config


def get_model_type(profile: str, path: str = None) -> str:
    """
    Return the model type from the configuration for the given profile.
    This function uses the new 'model' variable from the configuration.
    """
    config = get_profile_config(profile, path)
    if "model" not in config:
        raise KeyError(f"The 'model' variable is missing from configuration for profile '{profile}'")
    return config["model"]


def truncate_messages_by_token(messages: List[Dict[str, str]], max_tokens: int, model: str) -> List[Dict[str, str]]:
    if not messages:
        return messages
    
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

    # Calculate system prompt tokens if exists
    system_tokens = 0
    if system_prompt:
        system_tokens = len(enc.encode(system_prompt["content"])) + 3
        total += system_tokens

    # Process messages from newest to oldest
    for msg in reversed(messages):
        token_count = len(enc.encode(msg["content"])) + 3
        if total + token_count > max_tokens:
            break
        truncated.insert(0, msg)
        total += token_count

    # Ensure we have at least one message (excluding system prompt)
    if not truncated and messages:
        # If no messages fit, take the most recent one anyway
        last_msg = messages[-1]
        truncated = [last_msg]
        print(f"[Warning] Message truncation resulted in keeping only the most recent message. "
              f"Token count may exceed limit.")

    # Add system prompt back if it exists
    if system_prompt:
        truncated.insert(0, system_prompt)

    # Final safety check - ensure we never return empty array
    if not truncated and not system_prompt:
        print("[Error] All messages were truncated. This should not happen.")
        # Return a minimal user message to prevent API error
        return [{"role": "user", "content": "Hello"}]

    return truncated

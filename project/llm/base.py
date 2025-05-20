from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import json
import os
from datetime import datetime
from pathlib import Path


class LLMClientBase(ABC):
    """
    Base class for all LLM clients. Supports multi-turn chat by default.
    """

    def __init__(self, model_name: str, model: str):
        self.model_name = model_name
        self.default_model = model
        self.api_key = None
        self.messages: List[Dict[str, str]] = []  # full chat history

    @abstractmethod
    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        Submit a query to the LLM. If `image_path` is provided and supported,
        the implementation should handle it accordingly.
        """
        pass
    def set_system_prompt(self, system_prompt: str):
        """Optionally set system prompt message."""
        self.clear_history()
        self.messages.append({
            "role": "system",
            "content": system_prompt
        })

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "model": self.model_name  # optional metadata
        })

    def clear_history(self):
        """Clear all chat history."""
        self.messages.clear()

    def save_history(self, game_name: Optional[str] = None, filepath: Optional[str] = None):
        """
        Save chat history to a JSON file.
        If no filepath is provided, a log directory structure will be created as: log/{model_name}/{game_name}/
        and a timestamped filename will be used.
        """
        base_log_dir = Path(__file__).parent.parent / "log"
        try:
            base_log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            base_log_dir = Path(__file__).parent / "log"
            base_log_dir.mkdir(parents=True, exist_ok=True)

        if not filepath:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            filename = f"{timestamp}.json"
            subdir = base_log_dir / self.model_name
            if game_name:
                subdir = subdir / game_name
            subdir.mkdir(parents=True, exist_ok=True)
            filepath = subdir / filename

        with open(filepath, "w") as f:
            json.dump(self.messages, f, indent=2)

        print(f"[{self.model_name}] Chat history saved to {filepath}")

    def load_history(self, filepath: str):
        """Load chat history from a file."""
        with open(filepath, "r") as f:
            self.messages = json.load(f)

    def shutdown(self):
        """Optional resource cleanup."""
        pass

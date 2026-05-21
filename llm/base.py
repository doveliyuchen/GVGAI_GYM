from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import json
import os
from datetime import datetime
from pathlib import Path


class LLMClientBase(ABC):
    """Base class for all LLM clients with multi-turn chat support."""

    def __init__(self, model_name: str, model: str):
        self.model_name    = model_name
        self.default_model = model
        self.api_key: Optional[str] = None
        self.messages: List[Dict[str, str]] = []

        # Generation params (set externally by client.py via config)
        self.system_prompt: Optional[str] = None
        self.temperature:   Optional[float] = None
        self.max_tokens:    Optional[int]   = None
        self.top_p:         Optional[float] = None

    # ------------------------------------------------------------------ #
    # Abstract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Send a query; return the model's reply as a plain string."""

    # ------------------------------------------------------------------ #
    # Message management
    # ------------------------------------------------------------------ #

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self._rebuild_messages()

    def clear_history(self) -> None:
        """Clear conversation history, keeping the system prompt if set."""
        self._rebuild_messages()

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def _rebuild_messages(self) -> None:
        """Reset message list to [system prompt] (if any)."""
        self.messages.clear()
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    # ------------------------------------------------------------------ #
    # History persistence
    # ------------------------------------------------------------------ #

    def save_history(self, game_name: Optional[str] = None,
                     filepath: Optional[str] = None) -> None:
        output_path: Path
        if not filepath:
            base = Path(__file__).parent.parent / "log"
            base.mkdir(parents=True, exist_ok=True)
            subdir = base / self.model_name / (game_name or "")
            subdir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            output_path = subdir / f"{timestamp}.json"
        else:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.messages, f, indent=2)
        print(f"[{self.model_name}] Chat history saved to {output_path}")

    def load_history(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            self.messages = json.load(f)

    def shutdown(self) -> None:
        """Optional resource cleanup (override in subclasses)."""

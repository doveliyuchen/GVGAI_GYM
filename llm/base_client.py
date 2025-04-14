from abc import ABC, abstractmethod

# TODO: Add memory/logs?
class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self):
        """Initialize the base LLM client"""
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize client-specific configurations"""
        pass
    
    @abstractmethod
    def query(self, prompt: str) -> str:
        """Send a query to the LLM"""
        pass
    
    def __enter__(self):
        """Support for context manager protocol"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when used as a context manager"""
        self.shutdown()
    
    def shutdown(self) -> None:
        """Clean up resources (optional implementation in subclasses)"""
        pass

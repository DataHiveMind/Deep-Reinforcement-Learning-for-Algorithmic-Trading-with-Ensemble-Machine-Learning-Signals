from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """
    Abstract base class for all DRL agents.
    Defines the required interface for training, prediction, saving, and loading.
    """
    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the agent."""
        pass

    @abstractmethod
    def predict(self, state, *args, **kwargs) -> Any:
        """Predict an action given a state."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save the agent's parameters to a file."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load the agent's parameters from a file."""
        pass

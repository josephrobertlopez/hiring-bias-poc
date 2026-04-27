from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .data import Resume


class RuleProtocol(ABC):
    """Abstract interface for all skill rules."""

    @abstractmethod
    def fit(
        self, resumes: List[Resume], labels: List[bool]
    ) -> "RuleProtocol":
        """Learn rule patterns from training data."""
        pass

    @abstractmethod
    def matches(self, resume: Resume) -> bool:
        """Check if resume matches rule pattern."""
        pass

    @abstractmethod
    def score(self, resume: Resume) -> float:
        """Score resume match strength [0-1]."""
        pass

    @abstractmethod
    def explain(self, resume: Resume) -> Dict[str, Any]:
        """Explain why rule matched/didn't match."""
        pass

    @property
    @abstractmethod
    def rule_type(self) -> str:
        """Rule category: combination/experience/education/domain/gap/bias."""
        pass

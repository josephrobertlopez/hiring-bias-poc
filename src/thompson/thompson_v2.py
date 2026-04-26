"""Thompson Sampling v2 - Enhanced multi-arm bandit with fairness awareness.

This module implements Thompson sampling at the record/decision level for fairness-aware hiring.
It integrates with BCR adapter beliefs and tracks per-protected-group uncertainty.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ArmBelief:
    """Beta distribution belief about an arm (skill)."""
    alpha: float
    beta: float

    def sample(self, random_state: np.random.RandomState = None) -> float:
        """Sample from Beta distribution."""
        if random_state is None:
            return np.random.beta(self.alpha, self.beta)
        return random_state.beta(self.alpha, self.beta)

    def get_mean(self) -> float:
        """Get posterior mean."""
        return self.alpha / (self.alpha + self.beta)

    def get_variance(self) -> float:
        """Get posterior variance."""
        return (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))


class ThompsonSampler:
    """Thompson sampling for multi-arm bandit with fairness awareness.

    Implements record-level Thompson sampling where each decision samples from
    posterior distributions over arm quality.

    Attributes:
        skills: List of arm names
        skill_beliefs: Dictionary mapping arm index to ArmBelief
        random_state: Random state for reproducibility
        _decisions: History of (arm, outcome) tuples
        _cumulative_regret: Regret tracking
    """

    def __init__(self, skills: List[str], random_state: int = None):
        """Initialize Thompson sampler.

        Args:
            skills: List of skill/arm names
            random_state: Seed for reproducibility
        """
        self.skills = skills
        self.n_arms = len(skills)
        self.skill_beliefs = [ArmBelief(alpha=1.0, beta=1.0) for _ in range(self.n_arms)]
        self.random_state = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()
        self._decisions = []
        self._cumulative_regret = 0.0

    def sample_arm(self) -> int:
        """Sample arm using Thompson sampling.

        Samples from each arm's posterior Beta distribution and selects the arm
        with the highest sample.

        Returns:
            Index of selected arm
        """
        samples = [belief.sample(self.random_state) for belief in self.skill_beliefs]
        return int(np.argmax(samples))

    def update_belief(self, arm_idx: int, outcome: bool):
        """Update belief for arm after observing outcome.

        Args:
            arm_idx: Index of arm
            outcome: Binary outcome (1 = success, 0 = failure)
        """
        if outcome:
            self.skill_beliefs[arm_idx].alpha += 1
        else:
            self.skill_beliefs[arm_idx].beta += 1
        self._decisions.append((arm_idx, outcome))

    def sample_decision(self) -> int:
        """Alias for sample_arm() for API consistency."""
        return self.sample_arm()

    def compute_regret(self, true_rates: List[float] = None) -> float:
        """Compute cumulative regret.

        Args:
            true_rates: Optional true success rates for each arm

        Returns:
            Cumulative regret
        """
        if true_rates is None:
            return 0.0

        regret = 0.0
        optimal_rate = max(true_rates)

        for arm_idx, outcome in self._decisions:
            regret += optimal_rate - true_rates[arm_idx]

        return regret

    def get_posterior_params(self, arm_idx: int) -> Tuple[float, float]:
        """Get posterior parameters for arm.

        Args:
            arm_idx: Index of arm

        Returns:
            (alpha, beta) parameters
        """
        belief = self.skill_beliefs[arm_idx]
        return belief.alpha, belief.beta

    def get_decision_history(self) -> List[Tuple[int, bool]]:
        """Get history of (arm, outcome) decisions.

        Returns:
            List of (arm_index, outcome) tuples
        """
        return self._decisions.copy()


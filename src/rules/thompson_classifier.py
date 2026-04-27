from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .engine import SkillRulesEngine, SkillAuditResult
from .data import Resume, SkillVocabulary
from ..thompson.thompson_v2 import ThompsonSampler

@dataclass(frozen=True)
class ThompsonPrediction:
    prediction: int
    confidence: float
    rule_weights: Dict[str, float]
    exploration_bonus: float
    audit_result: SkillAuditResult
    regret_bound: float

class ThompsonRulesClassifier:
    def __init__(self, vocabulary: SkillVocabulary, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.rule_names = ['combination', 'experience', 'education', 'domain', 'gap', 'bias']
        self.rules_engine = SkillRulesEngine(vocabulary)
        self.thompson_sampler = ThompsonSampler(skills=self.rule_names)
        self.rule_performance: Dict[str, List[float]] = {name: [] for name in self.rule_names}
        self.fitted: bool = False
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def fit(self, resumes: List[Resume], labels: List[bool]) -> 'ThompsonRulesClassifier':
        self.rules_engine.fit(resumes, labels)
        for resume, label in zip(resumes, labels):
            audit_result = self.rules_engine.audit_resume(resume)
            for i, rule_name in enumerate(self.rule_names):
                rule_score = audit_result.rule_scores[rule_name]
                rule_pred = 1 if rule_score > 0.5 else 0
                outcome = 1 if rule_pred == label else 0
                self.thompson_sampler.update_belief(i, outcome)
                self.rule_performance[rule_name].append(float(outcome))
        self.fitted = True
        return self

    def predict(self, resume: Resume) -> ThompsonPrediction:
        """Predict with correct Thompson sampling regret bound.

        Regret bound: O(sqrt(K * t * log(t))) where K=number of arms, t=time steps
        """
        if not self.fitted:
            raise ValueError("Classifier must be fitted first")
        audit_result = self.rules_engine.audit_resume(resume)

        rule_weights = {}
        total_variance = 0.0
        for i, rule_name in enumerate(self.rule_names):
            alpha, beta = self.thompson_sampler.get_posterior_params(i)
            belief = self.thompson_sampler.skill_beliefs[i]
            sampled_weight = belief.get_mean()  # Use posterior mean instead of random sample
            rule_weights[rule_name] = sampled_weight
            total_variance += belief.get_variance()

        rule_scores = np.array([audit_result.rule_scores[name] for name in self.rule_names])
        weights_array = np.array([rule_weights[name] for name in self.rule_names])
        weighted_avg_score = np.dot(rule_scores, weights_array) / np.sum(weights_array) if np.sum(weights_array) > 0 else 0.5

        prediction = 1 if weighted_avg_score > 0.5 else 0
        confidence = abs(weighted_avg_score - 0.5) * 2
        exploration_bonus = total_variance / len(self.rule_names)

        # Correct Thompson sampling regret bound: O(sqrt(K * t * log(t)))
        # where K = number of arms, t = total number of decisions
        decisions = self.thompson_sampler.get_decision_history()
        t = len(decisions)  # Total number of time steps
        K = len(self.rule_names)  # Number of arms

        if t > 0:
            regret_bound = np.sqrt(K * t * np.log(max(t, 1)))
        else:
            regret_bound = 0.0

        return ThompsonPrediction(
            prediction=prediction,
            confidence=confidence,
            rule_weights=rule_weights,
            exploration_bonus=exploration_bonus,
            audit_result=audit_result,
            regret_bound=regret_bound
        )

    def predict_batch(self, resumes: List[Resume]) -> List[ThompsonPrediction]:
        if not self.fitted:
            raise ValueError("Classifier must be fitted first")
        return [self.predict(r) for r in resumes]

    def update_online(self, resume: Resume, true_label: int) -> None:
        if not self.fitted:
            raise ValueError("Classifier must be fitted first")
        audit_result = self.rules_engine.audit_resume(resume)
        for i, rule_name in enumerate(self.rule_names):
            rule_score = audit_result.rule_scores[rule_name]
            rule_pred = 1 if rule_score > 0.5 else 0
            outcome = 1 if rule_pred == true_label else 0
            self.thompson_sampler.update_belief(i, outcome)
            self.rule_performance[rule_name].append(float(outcome))

    def get_rule_rankings(self) -> List[Tuple[str, float, float]]:
        rankings = []
        for i, rule_name in enumerate(self.rule_names):
            alpha, beta = self.thompson_sampler.get_posterior_params(i)
            posterior_mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            posterior_std = np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))) if (alpha + beta + 1) > 0 else 0.0
            rankings.append((rule_name, posterior_mean, posterior_std))
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_exploration_stats(self) -> Dict[str, Any]:
        stats = {}
        for i, rule_name in enumerate(self.rule_names):
            rewards = self.rule_performance[rule_name]
            success_rate = sum(rewards) / len(rewards) if rewards else 0.0
            alpha, beta = self.thompson_sampler.get_posterior_params(i)
            stats[rule_name] = {
                'samples': len(rewards),
                'success_rate': success_rate,
                'alpha': alpha,
                'beta': beta
            }
        return stats

"""Core prediction logic with validation.

Handles all prediction operations with input validation
and structured output.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass

from ..data import Resume
from ..engine import SkillRulesEngine
from ..thompson_classifier import ThompsonRulesClassifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction output with metadata."""

    prediction: int  # 0/1 hire decision
    confidence: float  # [0-1] prediction confidence
    processing_time_ms: float
    bias_alerts: list  # Any bias warnings
    recommendations: list  # Skills to improve
    audit_summary: dict  # Key findings from audit
    model_version: str


class Predictor:
    """Handles prediction logic with input validation."""

    def __init__(
        self, engine: SkillRulesEngine, thompson: ThompsonRulesClassifier, version: str
    ):
        """Initialize predictor with fitted models.

        Args:
            engine: Fitted SkillRulesEngine instance
            thompson: Fitted ThompsonRulesClassifier instance
            version: Model version string
        """
        self.engine = engine
        self.thompson = thompson
        self.version = version

    def predict(self, resume: Resume, resume_id: str = "unknown") -> Optional[PredictionResult]:
        """Make validated prediction on a resume.

        Args:
            resume: Resume to predict on
            resume_id: Identifier for the resume

        Returns:
            PredictionResult if successful, None if validation fails
        """
        start_time = time.time()

        if not self._validate_resume(resume):
            logger.warning(f"Invalid resume: {resume_id}")
            return None

        try:
            # Get Thompson prediction (includes uncertainty and audit result)
            thompson_pred = self.thompson.predict(resume)

            # Get detailed audit from thompson prediction
            audit_result = thompson_pred.audit_result

            # Process bias alerts
            bias_alerts = [f"BIAS ALERT: {flag}" for flag in audit_result.bias_flags]

            # Processing time in milliseconds
            processing_time_ms = (time.time() - start_time) * 1000

            # Extract audit summary
            skill_patterns = (
                audit_result.skill_patterns[:3] if audit_result.skill_patterns else []
            )
            skill_gaps = audit_result.skill_gaps[:3] if audit_result.skill_gaps else []
            recommendations = (
                audit_result.recommendations[:5]
                if audit_result.recommendations
                else []
            )

            return PredictionResult(
                prediction=thompson_pred.prediction,
                confidence=thompson_pred.confidence,
                processing_time_ms=processing_time_ms,
                bias_alerts=bias_alerts,
                recommendations=recommendations,
                audit_summary={
                    "overall_score": audit_result.overall_score,
                    "rule_scores": audit_result.rule_scores,
                    "skill_patterns": skill_patterns,
                    "skill_gaps": skill_gaps,
                },
                model_version=self.version,
            )

        except Exception as e:
            logger.error(f"Prediction failed for {resume_id}: {e}")
            return None

    def _validate_resume(self, resume: Resume) -> bool:
        """Validate resume with specific type checks.

        Args:
            resume: Resume to validate

        Returns:
            True if all validations pass, False otherwise
        """
        try:
            return (
                isinstance(resume.skill_tokens, list)
                and len(resume.skill_tokens) > 0
                and isinstance(resume.years_experience, (int, float))
                and resume.years_experience >= 0
                and isinstance(resume.education_level, str)
                and len(resume.education_level.strip()) > 0
                and isinstance(resume.domain_background, list)
                and isinstance(resume.demographics, dict)
            )
        except Exception:
            return False

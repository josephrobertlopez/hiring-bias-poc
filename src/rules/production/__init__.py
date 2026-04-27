"""Production deployment components for SkillRulesEngine.

Exports:
- ModelStore: Model persistence with safer serialization
- Predictor: Core prediction logic with validation
- PredictionResult: Structured prediction output
- Monitor: Performance monitoring and health checks
- SkillRulesProduction: Production-ready deployment coordinator
- ProductionConfig: Configuration dataclass
- create_production_system: Factory function
- quick_predict: Quick prediction interface
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import traceback

from ..data import Resume, SkillVocabulary
from ..engine import SkillRulesEngine
from ..thompson_classifier import ThompsonRulesClassifier
from .model_store import ModelStore
from .predictor import Predictor, PredictionResult
from .monitor import Monitor, PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductionConfig:
    """Production deployment configuration."""

    model_path: str = "models/skill_rules_engine.pkl"
    vocabulary_path: str = "models/skill_vocabulary.json"
    log_predictions: bool = True
    enable_monitoring: bool = True
    batch_size: int = 100
    confidence_threshold: float = 0.7
    bias_alert_threshold: float = 0.8


class SkillRulesProduction:
    """Production-ready SkillRulesEngine deployment using focused components."""

    def __init__(self, config: ProductionConfig):
        """Initialize production system with focused components.

        Args:
            config: ProductionConfig instance
        """
        self.config = config
        self.model_store = ModelStore(config.model_path, config.vocabulary_path)
        self.monitor = Monitor()
        self.engine: Optional[SkillRulesEngine] = None
        self.thompson: Optional[ThompsonRulesClassifier] = None
        self.predictor: Optional[Predictor] = None
        self.vocabulary: Optional[SkillVocabulary] = None
        self.model_version = "1.0.0"
        self.is_loaded = False

    def load_model(self) -> bool:
        """Load trained models from disk.

        Returns:
            True if load successful, False otherwise
        """
        try:
            model_data = self.model_store.load()
            if model_data is None:
                return False

            self.engine = model_data["engine"]
            self.thompson = model_data["thompson"]
            self.vocabulary = model_data.get("vocabulary")
            self.model_version = model_data.get("version", "1.0.0")

            # Create predictor with loaded models
            self.predictor = Predictor(self.engine, self.thompson, self.model_version)
            self.is_loaded = True

            logger.info(f"Models loaded successfully (version {self.model_version})")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.error(traceback.format_exc())
            return False

    def save_model(
        self, engine: SkillRulesEngine, thompson: ThompsonRulesClassifier
    ) -> bool:
        """Save trained models to disk.

        Args:
            engine: Fitted SkillRulesEngine instance
            thompson: Fitted ThompsonRulesClassifier instance

        Returns:
            True if save successful, False otherwise
        """
        try:
            success = self.model_store.save(engine, thompson, self.model_version)
            if success:
                logger.info(f"Models saved successfully to {self.config.model_path}")
            return success

        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            logger.error(traceback.format_exc())
            return False

    def train(self, training_resumes: List[Resume], labels: List[bool]) -> bool:
        """Train the system on labeled data.

        Args:
            training_resumes: List of Resume instances
            labels: List of binary hiring labels

        Returns:
            True if training successful, False otherwise
        """
        try:
            if not training_resumes or not labels:
                logger.error("Empty training data")
                return False

            if len(training_resumes) != len(labels):
                logger.error(
                    f"Mismatch: {len(training_resumes)} resumes vs {len(labels)} labels"
                )
                return False

            logger.info(f"Training on {len(training_resumes)} resumes...")

            # Extract vocabulary from training resumes
            all_skills = set()
            all_domains = set()
            for resume in training_resumes:
                all_skills.update(resume.skill_tokens)
                all_domains.update(resume.domain_background)

            # Build vocabulary with skill categories
            self.vocabulary = SkillVocabulary(
                tokens=sorted(list(all_skills)),
                categories={"all_skills": list(all_skills), "domains": list(all_domains)},
                embeddings=None,
            )

            # Initialize models
            self.engine = SkillRulesEngine(self.vocabulary)
            self.thompson = ThompsonRulesClassifier(self.vocabulary)

            # Fit models
            self.engine.fit(training_resumes, labels)
            self.thompson.fit(training_resumes, labels)

            # Save trained models
            if not self.save_model(self.engine, self.thompson):
                return False

            self.is_loaded = True
            logger.info("Training completed successfully")
            logger.info(
                f"Vocabulary size: {len(self.vocabulary.tokens)} skills, "
                f"{len(all_domains)} domains"
            )
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            return False

    def predict(
        self, resume: Resume, resume_id: str = None
    ) -> Optional[PredictionResult]:
        """Make production prediction on a single resume.

        Args:
            resume: Resume to predict on
            resume_id: Optional identifier for the resume

        Returns:
            PredictionResult if successful, None otherwise
        """
        try:
            if self.predictor is None:
                if not self.load_model():
                    return None

            # Use predictor for coordinated prediction + monitoring
            result = self.predictor.predict(resume, resume_id or "unknown")

            # Record metrics
            if result:
                self.monitor.record_prediction(
                    result.processing_time_ms,
                    success=True,
                    has_bias_alert=len(result.bias_alerts) > 0,
                )

                # Log if enabled
                if self.config.log_predictions:
                    logger.info(
                        f"Prediction {resume_id}: {result.prediction} "
                        f"(conf={result.confidence:.3f})"
                    )
            else:
                self.monitor.record_prediction(0.0, success=False)

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {resume_id}: {e}")
            logger.error(traceback.format_exc())
            self.monitor.record_prediction(0.0, success=False)
            return None

    def predict_batch(
        self, resumes: List[Resume], resume_ids: Optional[List[str]] = None
    ) -> List[Optional[PredictionResult]]:
        """Batch prediction with performance optimization.

        Args:
            resumes: List of Resume instances
            resume_ids: Optional list of IDs (defaults to "resume_{i}")

        Returns:
            List of PredictionResult objects
        """
        if resume_ids is None:
            resume_ids = [f"resume_{i}" for i in range(len(resumes))]

        results = []
        batch_start = time.time()

        logger.info(f"Processing batch of {len(resumes)} resumes...")

        for i, resume in enumerate(resumes):
            result = self.predict(resume, resume_ids[i])
            results.append(result)

            # Progress logging
            if (i + 1) % self.config.batch_size == 0:
                elapsed = time.time() - batch_start
                logger.info(f"Processed {i+1}/{len(resumes)} resumes in {elapsed:.2f}s")

        total_time = time.time() - batch_start
        successful = sum(1 for r in results if r is not None)
        logger.info(
            f"Batch complete: {successful}/{len(resumes)} successful "
            f"in {total_time:.2f}s"
        )

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get system performance statistics.

        Returns:
            Dict with prediction counts, rates, and performance metrics
        """
        stats = self.monitor.get_stats()
        stats["model_version"] = self.model_version
        stats["is_loaded"] = self.is_loaded
        return stats

    def health_check(self) -> Dict[str, Any]:
        """System health check for monitoring.

        Returns:
            Dict with health status information
        """
        health = self.monitor.health_check()
        health["model_loaded"] = self.is_loaded
        health["model_version"] = self.model_version
        return health


def create_production_system(config_path: Optional[str] = None) -> SkillRulesProduction:
    """Create production system with configuration.

    Args:
        config_path: Optional path to JSON config file

    Returns:
        SkillRulesProduction instance
    """
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)
        config = ProductionConfig(**config_data)
    else:
        config = ProductionConfig()

    return SkillRulesProduction(config)


def quick_predict(
    resume_data: Dict[str, Any],
    model_path: str = "models/skill_rules_engine.pkl",
) -> Optional[Dict[str, Any]]:
    """Quick prediction interface for simple use cases.

    Args:
        resume_data: Dict with resume fields
        model_path: Path to model file

    Returns:
        Dict with prediction results or None if failed
    """
    try:
        # Create resume object
        resume = Resume(**resume_data)

        # Create system
        config = ProductionConfig(model_path=model_path)
        system = SkillRulesProduction(config)

        # Predict
        result = system.predict(resume)

        return asdict(result) if result else None

    except Exception as e:
        logger.error(f"Quick predict failed: {e}")
        logger.debug(traceback.format_exc())
        return None


__all__ = [
    "ModelStore",
    "Predictor",
    "PredictionResult",
    "Monitor",
    "PerformanceMetrics",
    "SkillRulesProduction",
    "ProductionConfig",
    "create_production_system",
    "quick_predict",
]

"""Model persistence with safer serialization.

Handles model storage and retrieval using joblib for safer
serialization than pickle.
"""

import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..data import SkillVocabulary
from ..engine import SkillRulesEngine
from ..thompson_classifier import ThompsonRulesClassifier

logger = logging.getLogger(__name__)


class ModelStore:
    """Handles model persistence with version control."""

    def __init__(self, model_path: str, vocabulary_path: str):
        """Initialize model store with paths.

        Args:
            model_path: Path to save/load model artifacts
            vocabulary_path: Path to save/load vocabulary
        """
        self.model_path = Path(model_path)
        self.vocabulary_path = Path(vocabulary_path)

    def save(
        self, engine: SkillRulesEngine, thompson: ThompsonRulesClassifier, version: str = "1.0.0"
    ) -> bool:
        """Save models using joblib (safer than pickle).

        Args:
            engine: Fitted SkillRulesEngine instance
            thompson: Fitted ThompsonRulesClassifier instance
            version: Model version string

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Create directories
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.vocabulary_path.parent.mkdir(parents=True, exist_ok=True)

            # Save vocabulary as JSON (safer and readable)
            vocab_data = {
                "tokens": engine.vocabulary.tokens,
                "categories": engine.vocabulary.categories,
            }
            with open(self.vocabulary_path, "w") as f:
                json.dump(vocab_data, f, indent=2)

            # Save models using joblib (safer than pickle)
            model_data = {
                "engine": engine,
                "thompson": thompson,
                "version": version,
                "joblib_version": joblib.__version__,
            }
            joblib.dump(model_data, self.model_path, compress=True)

            logger.info(f"Models saved successfully to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        """Load models from disk.

        Returns:
            Dict with 'engine', 'thompson', 'vocabulary', 'version' keys,
            or None if load fails
        """
        try:
            if not self.model_path.exists() or not self.vocabulary_path.exists():
                logger.error(f"Model files not found: {self.model_path}, {self.vocabulary_path}")
                return None

            # Load vocabulary from JSON
            with open(self.vocabulary_path, "r") as f:
                vocab_data = json.load(f)
            vocabulary = SkillVocabulary(
                tokens=vocab_data["tokens"],
                categories=vocab_data["categories"],
                embeddings=None,
            )

            # Load models from joblib
            model_data = joblib.load(self.model_path)
            model_data["vocabulary"] = vocabulary

            logger.info(f"Models loaded successfully from {self.model_path}")
            return model_data

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return None

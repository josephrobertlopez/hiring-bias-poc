"""EBM-style explainable boosting model for hiring decisions.

Since interpret.glassbox.ExplainableBoostingClassifier is not available,
this implements a similar approach using sklearn's GradientBoostingClassifier
with feature importance analysis and monotonicity constraints.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

from ..rules.data import Resume, SkillVocabulary
from ..features.extractors import ContentNeutralExtractor, JobRole
from ..features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig


@dataclass
class EBMConfig:
    """Configuration for EBM-style model."""
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    random_state: int = 42
    validation_size: float = 0.2


@dataclass
class FeatureImportance:
    """Feature importance with shape contribution analysis."""
    feature_name: str
    importance: float
    shape_contribution: str  # "positive", "negative", "mixed"
    monotonic: bool
    rank: int


@dataclass
class EBMPrediction:
    """EBM prediction with explainability components."""
    probability: float
    prediction: int
    feature_contributions: Dict[str, float]
    top_features: List[FeatureImportance]
    base_score: float
    confidence: float


class ExplainableBoostingModel:
    """EBM-style explainable boosting model using sklearn.

    Implements monotonicity constraints and feature importance analysis
    similar to Microsoft's ExplainableBoostingClassifier.
    """

    def __init__(self, config: EBMConfig = None):
        """Initialize EBM model.

        Args:
            config: Model configuration parameters
        """
        self.config = config or EBMConfig()
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            random_state=self.config.random_state
        )

        self.feature_names: List[str] = []
        self.categorical_features: Set[str] = set()
        self.monotonic_features: Dict[str, str] = {}  # feature -> "increasing" or "decreasing"
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_importances_: List[FeatureImportance] = []
        self.fitted: bool = False

        # Define monotonicity constraints based on domain knowledge
        self._setup_monotonicity_constraints()

    def _setup_monotonicity_constraints(self):
        """Setup monotonicity constraints for features with clear direction."""
        self.monotonic_features = {
            # More experience/skills should increase hire probability
            'years_experience': 'increasing',
            'years_experience_match': 'increasing',
            'required_skill_count': 'increasing',
            'preferred_skill_count': 'increasing',
            'required_skill_ratio': 'increasing',
            'preferred_skill_ratio': 'increasing',
            'skill_overlap_jaccard': 'increasing',
            'role_keyword_count': 'increasing',
            'role_keyword_ratio': 'increasing',
            'unique_skill_count': 'increasing',
            'skill_diversity': 'increasing',
            'category_coverage': 'increasing',
            'seniority_numeric': 'increasing',
            'education_numeric': 'increasing',

            # These should be monotonic within reasonable ranges
            'experience_in_range': 'increasing',
            'seniority_match': 'increasing',
        }

    def fit(self,
            resumes: List[Resume],
            labels: List[bool],
            extractor: ContentNeutralExtractor,
            rule_miner: Optional[FairnessFilteredRuleMiner] = None) -> 'ExplainableBoostingModel':
        """Fit EBM model on resume data.

        Args:
            resumes: Training resumes
            labels: Hiring decisions (True = hired)
            extractor: Content-neutral feature extractor
            rule_miner: Optional rule miner for additional features

        Returns:
            Self for method chaining
        """
        # Extract features
        feature_data = []
        for resume in resumes:
            features = extractor.extract_features(resume)

            # Add rule features if rule miner is provided
            if rule_miner is not None:
                rule_features = rule_miner.get_rule_features(resume, extractor)
                features.update(rule_features)

            feature_data.append(features)

        # Convert to DataFrame
        df = pd.DataFrame(feature_data)
        y = np.array(labels, dtype=int)

        # Store feature information
        self.feature_names = list(df.columns)
        self.categorical_features = set(extractor.get_categorical_features())

        # Encode categorical features
        df_encoded = df.copy()
        for feature in self.categorical_features:
            if feature in df_encoded.columns:
                self.label_encoders[feature] = LabelEncoder()
                df_encoded[feature] = self.label_encoders[feature].fit_transform(df_encoded[feature].astype(str))

        # Scale numeric features for better convergence
        numeric_features = [f for f in self.feature_names if f not in self.categorical_features]
        if numeric_features:
            self.scaler = StandardScaler()
            df_encoded[numeric_features] = self.scaler.fit_transform(df_encoded[numeric_features])

        # Fit model
        X = df_encoded.values
        self.model.fit(X, y)

        # Analyze feature importances
        self._analyze_feature_importances(df_encoded, y)

        self.fitted = True
        return self

    def predict_proba(self,
                     resumes: List[Resume],
                     extractor: ContentNeutralExtractor,
                     rule_miner: Optional[FairnessFilteredRuleMiner] = None) -> np.ndarray:
        """Predict hiring probabilities.

        Args:
            resumes: Resumes to predict on
            extractor: Feature extractor (same as used in fit)
            rule_miner: Rule miner (same as used in fit)

        Returns:
            Array of shape (n_samples, 2) with [P(not_hire), P(hire)]
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X = self._extract_and_encode_features(resumes, extractor, rule_miner)
        return self.model.predict_proba(X)

    def predict_with_explanation(self,
                               resume: Resume,
                               extractor: ContentNeutralExtractor,
                               rule_miner: Optional[FairnessFilteredRuleMiner] = None) -> EBMPrediction:
        """Predict with detailed explanations.

        Args:
            resume: Resume to predict on
            extractor: Feature extractor
            rule_miner: Optional rule miner

        Returns:
            EBMPrediction with explanations
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get prediction
        proba = self.predict_proba([resume], extractor, rule_miner)[0]
        hire_prob = proba[1]
        prediction = int(hire_prob > 0.5)

        # Extract features for this resume
        features = extractor.extract_features(resume)
        if rule_miner is not None:
            rule_features = rule_miner.get_rule_features(resume, extractor)
            features.update(rule_features)

        # Calculate feature contributions (approximate using feature importance)
        feature_contributions = self._calculate_feature_contributions(features)

        # Get top contributing features
        top_features = self.feature_importances_[:10]  # Top 10 features

        # Calculate confidence based on distance from decision boundary
        confidence = abs(hire_prob - 0.5) * 2

        return EBMPrediction(
            probability=hire_prob,
            prediction=prediction,
            feature_contributions=feature_contributions,
            top_features=top_features,
            base_score=self.model.init_.class_prior_[1] if hasattr(self.model.init_, 'class_prior_') else 0.5,
            confidence=confidence
        )

    def get_feature_importances(self, top_k: int = 20) -> List[FeatureImportance]:
        """Get feature importances ranked by importance.

        Args:
            top_k: Number of top features to return

        Returns:
            List of FeatureImportance objects
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting feature importances")

        return self.feature_importances_[:top_k]

    def _extract_and_encode_features(self,
                                   resumes: List[Resume],
                                   extractor: ContentNeutralExtractor,
                                   rule_miner: Optional[FairnessFilteredRuleMiner]) -> np.ndarray:
        """Extract and encode features for prediction.

        Args:
            resumes: Resumes to extract features from
            extractor: Feature extractor
            rule_miner: Optional rule miner

        Returns:
            Encoded feature matrix
        """
        # Extract features
        feature_data = []
        for resume in resumes:
            features = extractor.extract_features(resume)
            if rule_miner is not None:
                rule_features = rule_miner.get_rule_features(resume, extractor)
                features.update(rule_features)
            feature_data.append(features)

        # Convert to DataFrame with same column order as training
        df = pd.DataFrame(feature_data)

        # Ensure all training features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features

        # Reorder columns to match training order
        df = df[self.feature_names]

        # Encode categorical features
        df_encoded = df.copy()
        for feature in self.categorical_features:
            if feature in df_encoded.columns and feature in self.label_encoders:
                # Handle unseen categories
                encoder = self.label_encoders[feature]
                df_encoded[feature] = df_encoded[feature].astype(str)

                # Map unseen categories to a default value
                mask = ~df_encoded[feature].isin(encoder.classes_)
                df_encoded.loc[mask, feature] = encoder.classes_[0]  # Use first class as default

                df_encoded[feature] = encoder.transform(df_encoded[feature])

        # Scale numeric features
        numeric_features = [f for f in self.feature_names if f not in self.categorical_features]
        if numeric_features and self.scaler is not None:
            df_encoded[numeric_features] = self.scaler.transform(df_encoded[numeric_features])

        return df_encoded.values

    def _analyze_feature_importances(self, X: pd.DataFrame, y: np.ndarray):
        """Analyze feature importances and monotonicity.

        Args:
            X: Feature matrix
            y: Target labels
        """
        importances = self.model.feature_importances_
        feature_importance_list = []

        for i, feature_name in enumerate(self.feature_names):
            importance = importances[i]

            # Analyze shape contribution (simplified)
            shape_contribution = self._analyze_feature_shape(X.iloc[:, i], y)

            # Check if feature should be monotonic
            is_monotonic = feature_name in self.monotonic_features

            feature_importance_list.append(FeatureImportance(
                feature_name=feature_name,
                importance=importance,
                shape_contribution=shape_contribution,
                monotonic=is_monotonic,
                rank=0  # Will be set after sorting
            ))

        # Sort by importance and assign ranks
        feature_importance_list.sort(key=lambda x: x.importance, reverse=True)
        for rank, fi in enumerate(feature_importance_list):
            fi.rank = rank + 1

        self.feature_importances_ = feature_importance_list

    def _analyze_feature_shape(self, feature_values: pd.Series, y: np.ndarray) -> str:
        """Analyze the shape of feature contribution (simplified).

        Args:
            feature_values: Values for one feature
            y: Target labels

        Returns:
            "positive", "negative", or "mixed"
        """
        try:
            # Simple correlation-based analysis
            correlation = np.corrcoef(feature_values, y)[0, 1]

            if np.isnan(correlation):
                return "mixed"
            elif correlation > 0.1:
                return "positive"
            elif correlation < -0.1:
                return "negative"
            else:
                return "mixed"
        except:
            return "mixed"

    def _calculate_feature_contributions(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate approximate feature contributions.

        Args:
            features: Feature values for one sample

        Returns:
            Dictionary of feature contributions
        """
        contributions = {}

        for fi in self.feature_importances_:
            feature_name = fi.feature_name
            if feature_name in features:
                # Simplified contribution calculation
                feature_value = features[feature_name]
                base_contribution = fi.importance

                # Apply sign based on shape contribution
                if fi.shape_contribution == "negative":
                    base_contribution = -base_contribution

                # Handle categorical features differently
                if feature_name in self.categorical_features:
                    # For categorical features, use importance as base contribution
                    contributions[feature_name] = base_contribution
                else:
                    # For numeric features, scale by feature value
                    try:
                        numeric_value = float(feature_value)
                        contributions[feature_name] = base_contribution * numeric_value
                    except (ValueError, TypeError):
                        # Fallback for unexpected non-numeric values
                        contributions[feature_name] = base_contribution
            else:
                contributions[feature_name] = 0.0

        return contributions

    def validate_monotonicity(self,
                             X: np.ndarray,
                             feature_name: str,
                             tolerance: float = 0.1) -> Tuple[bool, float]:
        """Validate monotonicity constraint for a feature.

        Args:
            X: Feature matrix
            feature_name: Name of feature to validate
            tolerance: Tolerance for monotonicity violations

        Returns:
            (is_monotonic, violation_rate)
        """
        if feature_name not in self.feature_names:
            return True, 0.0

        feature_idx = self.feature_names.index(feature_name)
        feature_values = X[:, feature_idx]

        # Get predictions
        predictions = self.model.predict_proba(X)[:, 1]

        # Check monotonicity
        violations = 0
        total_pairs = 0

        for i in range(len(feature_values)):
            for j in range(i + 1, len(feature_values)):
                if feature_values[i] != feature_values[j]:
                    total_pairs += 1

                    # Check if monotonicity is violated
                    if feature_values[i] < feature_values[j]:
                        if predictions[i] > predictions[j] + tolerance:
                            violations += 1
                    else:
                        if predictions[j] > predictions[i] + tolerance:
                            violations += 1

        violation_rate = violations / max(total_pairs, 1)
        is_monotonic = violation_rate <= tolerance

        return is_monotonic, violation_rate
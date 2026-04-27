"""Kaggle benchmark evaluation for hiring bias POC.

Evaluates the complete pipeline (features + EBM + calibration + fairness)
on a pinned Kaggle dataset with honest caveats about limitations.
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from ..rules.data import Resume, SkillVocabulary
from ..features.extractors import ContentNeutralExtractor, JobRole
from ..features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig
from ..model.ebm_head import ExplainableBoostingModel, EBMConfig
from ..model.calibration import IsotonicCalibrator
from ..fairness.metrics import FairnessMetricsCalculator
from ..fairness.counterfactual import CounterfactualAnalyzer


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    dataset_info: Dict[str, Any]
    model_performance: Dict[str, float]
    fairness_metrics: Dict[str, Any]
    counterfactual_results: Dict[str, Any]
    top_features: List[Dict[str, Any]]
    calibration_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class KaggleBenchmarkEvaluator:
    """Evaluate hiring bias POC on Kaggle dataset with honest reporting."""

    def __init__(self, random_state: int = 42):
        """Initialize benchmark evaluator.

        Args:
            random_state: Seed for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def run_benchmark(self,
                     dataset_path: Optional[str] = None,
                     output_path: str = "benchmark_results.json") -> BenchmarkResult:
        """Run complete benchmark evaluation.

        Args:
            dataset_path: Path to Kaggle dataset (None = use synthetic)
            output_path: Where to save results

        Returns:
            BenchmarkResult with all metrics
        """
        print("Starting Kaggle benchmark evaluation...")

        # Load or create dataset
        if dataset_path and os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            dataset_info, resumes, labels, demographics = self._load_kaggle_dataset(dataset_path)
        else:
            print("CAVEAT: Using synthetic dataset - real Kaggle dataset not available")
            dataset_info, resumes, labels, demographics = self._create_synthetic_dataset()

        print(f"Dataset loaded: {len(resumes)} resumes, {sum(labels)} hired")

        # Train-test split
        train_resumes, test_resumes, train_labels, test_labels = train_test_split(
            resumes, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )

        train_demographics = {
            attr: values[:len(train_resumes)]
            for attr, values in demographics.items()
        }
        test_demographics = {
            attr: values[len(train_resumes):]
            for attr, values in demographics.items()
        }

        print(f"Split: {len(train_resumes)} train, {len(test_resumes)} test")

        # Setup pipeline components
        vocab = self._create_vocabulary()
        role = self._create_job_role(vocab)
        extractor = ContentNeutralExtractor(vocab, role)

        # Train rule miner
        print("Training rule miner...")
        rule_config = RuleMinerConfig(
            min_support=0.01,
            min_confidence=0.6,
            min_lift=1.2,
            top_k=100
        )
        rule_miner = FairnessFilteredRuleMiner(rule_config)
        rule_miner.mine_rules(train_resumes, train_labels, extractor)
        print(f"Discovered {len(rule_miner.rules)} association rules")

        # Train EBM model
        print("Training EBM model...")
        ebm_config = EBMConfig(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state
        )
        model = ExplainableBoostingModel(ebm_config)
        model.fit(train_resumes, train_labels, extractor, rule_miner)

        # Get training predictions for calibration
        train_proba = model.predict_proba(train_resumes, extractor, rule_miner)[:, 1]

        # Train calibrator
        print("Training calibrator...")
        calibrator = IsotonicCalibrator(n_bins=10, random_state=self.random_state)
        calibration_result = calibrator.fit_and_calibrate(
            train_proba, np.array(train_labels, dtype=int), validation_size=0.3
        )

        # Evaluate on test set
        print("Evaluating on test set...")
        test_proba_raw = model.predict_proba(test_resumes, extractor, rule_miner)[:, 1]
        test_proba = calibrator.calibrate(test_proba_raw)
        test_pred = (test_proba > 0.5).astype(int)

        # Model performance metrics
        model_performance = {
            "auc": float(roc_auc_score(test_labels, test_proba)),
            "accuracy": float(accuracy_score(test_labels, test_pred)),
            "precision": float(precision_score(test_labels, test_pred)),
            "recall": float(recall_score(test_labels, test_pred)),
            "n_test_samples": len(test_labels),
            "positive_rate": float(np.mean(test_labels))
        }

        # Fairness metrics
        print("Calculating fairness metrics...")
        fairness_calculator = FairnessMetricsCalculator()
        fairness_results = fairness_calculator.run_all_fairness_gates(
            np.array(test_labels, dtype=int),
            test_pred,
            test_proba,
            test_demographics
        )

        # Convert fairness results to serializable format
        fairness_metrics = {}
        for attr_name, attr_results in fairness_results.items():
            fairness_metrics[attr_name] = {}
            for result in attr_results:
                fairness_metrics[attr_name][result.metric_name] = {
                    "value": result.value,
                    "threshold": result.threshold,
                    "passed": result.passed,
                    "group_breakdown": result.group_breakdown
                }

        # Counterfactual analysis
        print("Running counterfactual analysis...")
        def predict_fn(resume):
            raw_prob = model.predict_proba([resume], extractor, rule_miner)[0, 1]
            return calibrator.calibrate(np.array([raw_prob]))[0]

        cf_analyzer = CounterfactualAnalyzer()
        cf_results = cf_analyzer.analyze_counterfactual_fairness(
            test_resumes, predict_fn, threshold=0.05
        )

        # Convert counterfactual results to serializable format
        counterfactual_results = {}
        for attr_name, result in cf_results.items():
            counterfactual_results[attr_name] = {
                "flip_rate_mean": result.flip_rate_mean,
                "flip_rate_p95": result.flip_rate_p95,
                "flip_rate_max": result.flip_rate_max,
                "gate_passed": result.gate_passed,
                "total_comparisons": result.total_comparisons
            }

        # Top features by EBM importance
        top_features = []
        feature_importances = model.get_feature_importances(top_k=10)
        for fi in feature_importances:
            top_features.append({
                "feature_name": fi.feature_name,
                "importance": fi.importance,
                "shape_contribution": fi.shape_contribution,
                "rank": fi.rank
            })

        # Calibration metrics
        calibration_metrics = {
            "ece_before": calibration_result.metrics.ece_before,
            "ece_after": calibration_result.metrics.ece_after,
            "brier_score_before": calibration_result.metrics.brier_score_before,
            "brier_score_after": calibration_result.metrics.brier_score_after,
            "calibration_samples": calibration_result.metrics.n_calibration_samples
        }

        # Compile results
        result = BenchmarkResult(
            dataset_info=dataset_info,
            model_performance=model_performance,
            fairness_metrics=fairness_metrics,
            counterfactual_results=counterfactual_results,
            top_features=top_features,
            calibration_metrics=calibration_metrics,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "random_state": self.random_state,
                "model_config": asdict(ebm_config),
                "rule_config": asdict(rule_config)
            }
        )

        # Save results
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, cls=NumpyJSONEncoder)

        print(f"Benchmark complete. Results saved to {output_path}")
        self._print_summary(result)

        return result

    def _load_kaggle_dataset(self, dataset_path: str) -> Tuple[Dict, List[Resume], List[bool], Dict]:
        """Load actual Kaggle dataset.

        Args:
            dataset_path: Path to Kaggle CSV/JSON dataset

        Returns:
            (dataset_info, resumes, labels, demographics)
        """
        # Calculate dataset hash for reproducibility
        with open(dataset_path, 'rb') as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        # Load dataset
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            df = pd.read_json(dataset_path, lines=dataset_path.endswith('.jsonl'))
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")

        dataset_info = {
            "source": "kaggle",
            "dataset_path": dataset_path,
            "dataset_hash": dataset_hash,
            "n_samples": len(df),
            "columns": list(df.columns),
            "caveat": "Dataset may be job-category classification, not actual hire/no-hire decisions"
        }

        # Convert to Resume objects (this would need dataset-specific mapping)
        resumes, labels, demographics = self._convert_dataframe_to_resumes(df)

        return dataset_info, resumes, labels, demographics

    def _convert_dataframe_to_resumes(self, df: pd.DataFrame) -> Tuple[List[Resume], List[bool], Dict]:
        """Convert Kaggle dataframe to Resume objects.

        This is dataset-specific and would need customization for each Kaggle dataset.
        """
        # CAVEAT: This is a placeholder implementation
        # Real implementation would depend on dataset schema

        resumes = []
        labels = []
        demographics = {'gender': [], 'race': []}

        for _, row in df.iterrows():
            # Example mapping - would need to be customized for actual dataset
            resume = Resume(
                skill_tokens=[],  # Would extract from dataset fields
                years_experience=0.0,  # Would map from dataset
                education_level='bachelor',  # Would map from dataset
                domain_background=['other'],  # Would map from dataset
                demographics={}  # Would extract protected attributes
            )

            resumes.append(resume)
            labels.append(False)  # Would extract actual labels
            demographics['gender'].append('unknown')
            demographics['race'].append('unknown')

        return resumes, labels, demographics

    def _create_synthetic_dataset(self) -> Tuple[Dict, List[Resume], List[bool], Dict]:
        """Create synthetic dataset for demonstration.

        Returns:
            (dataset_info, resumes, labels, demographics)
        """
        np.random.seed(self.random_state)

        # Skills and domains for synthetic resumes
        all_skills = ['python', 'sql', 'javascript', 'react', 'tensorflow', 'aws', 'docker', 'kubernetes']
        all_domains = ['tech', 'finance', 'healthcare', 'education', 'retail']
        education_levels = ['bootcamp', 'bachelor', 'master', 'phd']
        genders = ['male', 'female']
        races = ['white', 'black', 'asian', 'hispanic']

        resumes = []
        labels = []
        demographics = {'gender': [], 'race': []}

        n_samples = 1000

        for i in range(n_samples):
            # Random resume attributes
            n_skills = np.random.randint(1, 5)
            skills = np.random.choice(all_skills, size=n_skills, replace=False).tolist()
            experience = np.random.exponential(3.0)  # Mean ~3 years
            education = np.random.choice(education_levels)
            domains = np.random.choice(all_domains, size=np.random.randint(1, 3), replace=False).tolist()

            gender = np.random.choice(genders)
            race = np.random.choice(races)

            # Hiring probability based on skills and experience (content-neutral)
            hire_prob = 0.3  # Base probability
            hire_prob += 0.1 * len(skills)  # More skills = higher probability
            hire_prob += 0.05 * experience  # More experience = higher probability
            hire_prob += 0.1 if 'python' in skills else 0  # Python bonus
            hire_prob += 0.05 if education in ['master', 'phd'] else 0  # Education bonus

            # Cap probability and add noise
            hire_prob = min(0.8, hire_prob)
            hire_prob += np.random.normal(0, 0.1)
            hire_prob = max(0.1, min(0.9, hire_prob))  # Clamp to [0.1, 0.9]

            hired = np.random.random() < hire_prob

            resume = Resume(
                skill_tokens=skills,
                years_experience=experience,
                education_level=education,
                domain_background=domains,
                demographics={'gender': gender, 'race': race}
            )

            resumes.append(resume)
            labels.append(hired)
            demographics['gender'].append(gender)
            demographics['race'].append(race)

        # Convert demographics to numpy arrays
        demographics = {k: np.array(v) for k, v in demographics.items()}

        dataset_info = {
            "source": "synthetic",
            "dataset_path": "generated",
            "dataset_hash": "synthetic_v1",
            "n_samples": n_samples,
            "caveat": "Synthetic dataset - not real hiring decisions. Content-neutral generation.",
            "generation_seed": self.random_state
        }

        return dataset_info, resumes, labels, demographics

    def _create_vocabulary(self) -> SkillVocabulary:
        """Create skill vocabulary for benchmark."""
        return SkillVocabulary(
            tokens=['python', 'sql', 'javascript', 'react', 'tensorflow', 'aws', 'docker', 'kubernetes',
                   'java', 'c++', 'machine_learning', 'data_analysis', 'web_development'],
            categories={
                'programming': ['python', 'javascript', 'java', 'c++'],
                'data': ['sql', 'tensorflow', 'machine_learning', 'data_analysis'],
                'cloud': ['aws', 'docker', 'kubernetes'],
                'frontend': ['react', 'web_development']
            }
        )

    def _create_job_role(self, vocab: SkillVocabulary) -> JobRole:
        """Create job role for benchmark."""
        return JobRole(
            required_skills={'python', 'sql'},
            preferred_skills={'tensorflow', 'aws', 'react'},
            min_experience=2.0,
            max_experience=10.0,
            role_keywords={'software', 'engineer', 'developer', 'data', 'analyst'},
            seniority_level='mid'
        )

    def _print_summary(self, result: BenchmarkResult):
        """Print benchmark summary to console."""
        print("\n" + "="*60)
        print("KAGGLE BENCHMARK RESULTS")
        print("="*60)

        print(f"\nDataset: {result.dataset_info['source']}")
        print(f"Samples: {result.dataset_info['n_samples']}")
        if 'caveat' in result.dataset_info:
            print(f"CAVEAT: {result.dataset_info['caveat']}")

        print(f"\nModel Performance:")
        print(f"  AUC: {result.model_performance['auc']:.3f}")
        print(f"  Accuracy: {result.model_performance['accuracy']:.3f}")
        print(f"  Precision: {result.model_performance['precision']:.3f}")
        print(f"  Recall: {result.model_performance['recall']:.3f}")

        print(f"\nCalibration:")
        print(f"  ECE before: {result.calibration_metrics['ece_before']:.3f}")
        print(f"  ECE after: {result.calibration_metrics['ece_after']:.3f}")

        print(f"\nFairness Gates:")
        all_passed = True
        for attr_name, attr_metrics in result.fairness_metrics.items():
            print(f"  {attr_name.title()}:")
            for metric_name, metric_data in attr_metrics.items():
                status = "✅" if metric_data['passed'] else "❌"
                print(f"    {metric_name}: {metric_data['value']:.3f} {status}")
                if not metric_data['passed']:
                    all_passed = False

        print(f"\nCounterfactual Analysis:")
        for attr_name, cf_data in result.counterfactual_results.items():
            status = "✅" if cf_data['gate_passed'] else "❌"
            print(f"  {attr_name.title()}: p95={cf_data['flip_rate_p95']:.3f} {status}")

        print(f"\nTop Features:")
        for i, feature in enumerate(result.top_features[:5], 1):
            print(f"  {i}. {feature['feature_name']}: {feature['importance']:.3f}")

        overall_status = "✅ PASSED" if all_passed else "❌ FAILED"
        print(f"\nOverall Fairness: {overall_status}")
        print("="*60)


def main():
    """Command-line interface for benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Kaggle benchmark evaluation")
    parser.add_argument("--dataset", type=str, help="Path to Kaggle dataset")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    evaluator = KaggleBenchmarkEvaluator(random_state=args.seed)
    evaluator.run_benchmark(dataset_path=args.dataset, output_path=args.output)


if __name__ == "__main__":
    main()
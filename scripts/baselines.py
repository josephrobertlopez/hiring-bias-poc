"""AUC baseline diagnosis script.

Runs 4 pipelines on identical train/test data to diagnose the 0.649 AUC bottleneck:
- stratified_random: no features (~0.50 floor)
- logreg_raw: content-neutral features only (feature quality probe)
- ebm_raw: content-neutral features only (EBM vs logreg on same features)
- ebm_full: content-neutral + rule features (current pipeline)

Purpose: determine if bottleneck is features, EBM training, or rule pipeline.
"""

import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
from interpret.glassbox import ExplainableBoostingClassifier

# Import existing modules (read-only diagnostic)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rules.data import Resume, SkillVocabulary
from src.features.extractors import ContentNeutralExtractor, JobRole
from src.features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig
from src.benchmarks.kaggle_eval import NumpyJSONEncoder


class BaselineDiagnostic:
    """Diagnostic to identify AUC bottleneck."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_diagnosis(self, output_path: str = "baselines.json") -> Dict[str, Any]:
        """Run 4-pipeline diagnosis and save results."""
        print("Starting AUC baseline diagnosis...")
        print("=" * 60)

        # 1. Load same dataset as kaggle_eval.py
        dataset_info, resumes, labels, demographics = self._create_synthetic_dataset()
        print(f"Dataset loaded: {len(resumes)} resumes, {sum(labels)} hired")

        # 2. Same train/test split as kaggle_eval.py
        train_resumes, test_resumes, train_labels, test_labels = train_test_split(
            resumes, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        print(f"Split: {len(train_resumes)} train, {len(test_resumes)} test")

        # 3. Setup feature extractors (reuse existing)
        vocab = SkillVocabulary(
            tokens=['python', 'sql', 'javascript', 'react', 'tensorflow', 'aws', 'docker', 'kubernetes'],
            categories={
                'programming': ['python', 'javascript'],
                'data': ['sql', 'tensorflow'],
                'cloud': ['aws', 'docker', 'kubernetes'],
                'frontend': ['react', 'javascript']
            }
        )
        role = JobRole(
            required_skills={'python', 'sql'},
            preferred_skills={'javascript', 'react', 'tensorflow'},
            min_experience=2.0,
            max_experience=8.0,
            role_keywords={'software', 'engineer', 'developer', 'programming'},
            seniority_level='mid'
        )
        extractor = ContentNeutralExtractor(vocab, role)

        # 4. Run 4 pipelines
        results = {}

        print("Running baselines...")
        results['stratified_random'] = self._run_stratified_random(test_labels)
        results['logreg_raw'] = self._run_logreg_raw(train_resumes, test_resumes, train_labels, test_labels, extractor)
        results['ebm_raw'] = self._run_ebm_raw(train_resumes, test_resumes, train_labels, test_labels, extractor)
        results['ebm_full'] = self._run_ebm_full(train_resumes, test_resumes, train_labels, test_labels, extractor)

        # 5. Print diagnosis table
        self._print_diagnosis_table(results)

        # 6. Save results
        baseline_data = {
            "dataset_info": dataset_info,
            "baseline_results": results,
            "diagnosis": self._generate_diagnosis(results),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "random_state": self.random_state
            }
        }

        with open(output_path, 'w') as f:
            json.dump(baseline_data, f, indent=2, cls=NumpyJSONEncoder)

        print(f"\nResults saved to {output_path}")
        return baseline_data

    def _create_synthetic_dataset(self) -> Tuple[Dict, List[Resume], List[bool], Dict]:
        """Create synthetic dataset - same as kaggle_eval.py."""
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

    def _run_stratified_random(self, test_labels: List[bool]) -> Dict[str, Any]:
        """Baseline 1: Stratified random (floor performance ~0.50)."""
        dummy = DummyClassifier(strategy='stratified', random_state=self.random_state)
        dummy.fit(np.zeros((len(test_labels), 1)), test_labels)  # Dummy features
        y_pred = dummy.predict_proba(np.zeros((len(test_labels), 1)))[:, 1]

        auc = roc_auc_score(test_labels, y_pred)

        return {
            "auc": auc,
            "features": "none",
            "notes": "theoretical floor",
            "top_features": []
        }

    def _run_logreg_raw(self, train_resumes, test_resumes, train_labels, test_labels, extractor) -> Dict[str, Any]:
        """Baseline 2: Logistic regression on content-neutral features only."""
        # Extract features
        X_train = self._extract_feature_matrix(train_resumes, extractor)
        X_test = self._extract_feature_matrix(test_resumes, extractor)

        # Train logistic regression
        logreg = LogisticRegression(max_iter=1000, random_state=self.random_state)
        logreg.fit(X_train, train_labels)

        # Predict and evaluate
        y_pred = logreg.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(test_labels, y_pred)

        # Get feature importances (coefficients)
        feature_names = list(X_train.columns)
        importances = np.abs(logreg.coef_[0])
        top_features = self._get_top_features(feature_names, importances, top_k=5)

        return {
            "auc": auc,
            "features": "content-neutral only",
            "notes": "feature quality probe",
            "top_features": top_features
        }

    def _run_ebm_raw(self, train_resumes, test_resumes, train_labels, test_labels, extractor) -> Dict[str, Any]:
        """Baseline 3: EBM on content-neutral features only."""
        # Extract features
        X_train = self._extract_feature_matrix(train_resumes, extractor)
        X_test = self._extract_feature_matrix(test_resumes, extractor)

        # Train EBM
        ebm = ExplainableBoostingClassifier(random_state=self.random_state)
        ebm.fit(X_train, train_labels)

        # Predict and evaluate
        y_pred = ebm.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(test_labels, y_pred)

        # Get feature importances
        feature_names = list(X_train.columns)
        importances = np.array(ebm.term_importances())
        top_features = self._get_top_features(feature_names, importances, top_k=5)

        return {
            "auc": auc,
            "features": "content-neutral only",
            "notes": "EBM vs logreg on same features",
            "top_features": top_features
        }

    def _run_ebm_full(self, train_resumes, test_resumes, train_labels, test_labels, extractor) -> Dict[str, Any]:
        """Baseline 4: EBM on content-neutral + rule features (current pipeline)."""
        # Extract base features
        X_train_base = self._extract_feature_matrix(train_resumes, extractor)
        X_test_base = self._extract_feature_matrix(test_resumes, extractor)

        # Add rule features using existing rule miner
        rule_config = RuleMinerConfig(min_support=0.01, min_confidence=0.6, min_lift=1.2, top_k=100)
        rule_miner = FairnessFilteredRuleMiner(rule_config)
        rule_miner.mine_rules(train_resumes, train_labels, extractor)

        # Get rule features for train/test
        X_train_rules = self._extract_rule_features(train_resumes, rule_miner, extractor)
        X_test_rules = self._extract_rule_features(test_resumes, rule_miner, extractor)

        # Combine base + rule features
        X_train_combined = self._combine_features(X_train_base, X_train_rules)
        X_test_combined = self._combine_features(X_test_base, X_test_rules)

        # Train EBM
        ebm = ExplainableBoostingClassifier(random_state=self.random_state)
        ebm.fit(X_train_combined, train_labels)

        # STEP 6 DIAGNOSTIC: Print config for reconciliation with kaggle_eval.py
        print(f"\n=== BASELINES.PY EBM_FULL DIAGNOSTIC ===")
        print(f"Train/test split: {len(train_resumes)} train, {len(test_resumes)} test")
        print(f"Number of features: {len(X_train_combined.columns)}")
        print(f"Feature names: {sorted(X_train_combined.columns.tolist())}")
        print(f"EBM config: random_state={self.random_state} (interpret defaults for others)")
        print(f"Random state at fit: {self.random_state}")
        print(f"Rule miner provided: True")
        print(f"=======================================")

        # Predict and evaluate
        y_pred = ebm.predict_proba(X_test_combined)[:, 1]
        auc = roc_auc_score(test_labels, y_pred)

        # Get feature importances
        feature_names = list(X_train_combined.columns)
        importances = np.array(ebm.term_importances())
        top_features = self._get_top_features(feature_names, importances, top_k=5)

        return {
            "auc": auc,
            "features": "content-neutral + rule features",
            "notes": "current pipeline",
            "top_features": top_features
        }

    def _extract_feature_matrix(self, resumes, extractor):
        """Extract content-neutral features as pandas DataFrame."""
        import pandas as pd

        feature_dicts = [extractor.extract_features(resume) for resume in resumes]
        df = pd.DataFrame(feature_dicts)

        # Filter to only numeric and binary features (exclude categorical strings)
        numeric_features = extractor.get_numeric_features()
        binary_features = extractor.get_binary_features()
        allowed_features = numeric_features + binary_features

        # Only keep columns that are in the allowed features and exist in the DataFrame
        available_features = [f for f in allowed_features if f in df.columns]
        return df[available_features]

    def _extract_rule_features(self, resumes, rule_miner, extractor):
        """Extract rule features as pandas DataFrame."""
        import pandas as pd

        rule_features = []
        for resume in resumes:
            # Get rule features directly from rule miner
            features = rule_miner.get_rule_features(resume, extractor)
            rule_features.append(features)

        return pd.DataFrame(rule_features)

    def _combine_features(self, base_features, rule_features):
        """Combine base and rule features."""
        import pandas as pd

        return pd.concat([base_features, rule_features], axis=1)

    def _get_top_features(self, feature_names, importances, top_k=5):
        """Get top-k features by importance."""
        indices = np.argsort(importances)[::-1][:top_k]
        return [
            {"feature_name": feature_names[i], "importance": float(importances[i]), "rank": rank + 1}
            for rank, i in enumerate(indices)
        ]

    def _print_diagnosis_table(self, results):
        """Print diagnosis table."""
        print("\n" + "=" * 80)
        print("AUC BASELINE DIAGNOSIS")
        print("=" * 80)
        print(f"{'Pipeline':<20} {'Features':<25} {'Test AUC':<10} {'Notes':<20}")
        print("-" * 80)

        for pipeline, result in results.items():
            auc = f"{result['auc']:.3f}"
            print(f"{pipeline:<20} {result['features']:<25} {auc:<10} {result['notes']:<20}")

        print("-" * 80)

        # Print top features for each pipeline
        for pipeline, result in results.items():
            if result['top_features']:
                print(f"\nTop-5 features ({pipeline}):")
                for feat in result['top_features']:
                    print(f"  {feat['rank']}. {feat['feature_name']}: {feat['importance']:.3f}")

        print("=" * 80)

    def _generate_diagnosis(self, results) -> str:
        """Generate one-paragraph diagnosis."""
        auc_random = results['stratified_random']['auc']
        auc_logreg = results['logreg_raw']['auc']
        auc_ebm_raw = results['ebm_raw']['auc']
        auc_ebm_full = results['ebm_full']['auc']

        # Diagnosis logic
        if abs(auc_logreg - auc_ebm_full) < 0.02:  # ~same performance
            return f"feature bottleneck: logreg_raw ({auc_logreg:.3f}) ≈ ebm_full ({auc_ebm_full:.3f}) suggests features are the limiting factor, not model choice or rules"
        elif auc_ebm_raw > auc_logreg + 0.05 and abs(auc_ebm_full - auc_ebm_raw) < 0.02:
            return f"rule features redundant: ebm_raw ({auc_ebm_raw:.3f}) >> logreg_raw ({auc_logreg:.3f}) but ebm_full ({auc_ebm_full:.3f}) ≈ ebm_raw suggests rule features add no signal"
        elif auc_ebm_full < auc_ebm_raw - 0.02:
            return f"rule features hurting: ebm_full ({auc_ebm_full:.3f}) < ebm_raw ({auc_ebm_raw:.3f}) suggests fairness filter is removing predictive signal"
        elif auc_logreg > 0.75 and auc_ebm_full < 0.70:
            return f"EBM bug: logreg_raw ({auc_logreg:.3f}) >> ebm_full ({auc_ebm_full:.3f}) suggests EBM pipeline has calibration/scoring issues"
        else:
            return f"mixed signals: logreg={auc_logreg:.3f}, ebm_raw={auc_ebm_raw:.3f}, ebm_full={auc_ebm_full:.3f} - multiple factors may be involved"


def run_multi_seed_analysis(base_seed: int, n_seeds: int, output_path: str) -> Dict[str, Any]:
    """Run baseline analysis across multiple seeds for noise bracketing."""
    seeds = list(range(base_seed, base_seed + n_seeds))
    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\nSeed {i+1}/{n_seeds}: {seed}")
        diagnostic = BaselineDiagnostic(random_state=seed)

        # Run quietly (no individual prints)
        temp_output = f"temp_baselines_{seed}.json"
        result = diagnostic.run_diagnosis(temp_output)
        all_results.append(result)

        # Clean up temp file
        import os
        if os.path.exists(temp_output):
            os.remove(temp_output)

    # Compute statistics
    pipelines = ['stratified_random', 'logreg_raw', 'ebm_raw', 'ebm_full']
    multi_seed_stats = {}

    for pipeline in pipelines:
        aucs = [result['baseline_results'][pipeline]['auc'] for result in all_results]
        multi_seed_stats[pipeline] = {
            'mean': np.mean(aucs),
            'std': np.std(aucs),
            'min': np.min(aucs),
            'max': np.max(aucs),
            'aucs': aucs
        }

    # Save results with multi_seed_results key
    final_result = all_results[0]  # Use first result as template
    final_result['multi_seed_results'] = {
        'seeds': seeds,
        'n_seeds': n_seeds,
        'pipeline_stats': multi_seed_stats
    }

    with open(output_path, 'w') as f:
        json.dump(final_result, f, indent=2, cls=NumpyJSONEncoder)

    print(f"\nMulti-seed results saved to {output_path}")
    return final_result


def print_multi_seed_summary(results: Dict[str, Any]):
    """Print summary of multi-seed analysis."""
    stats = results['multi_seed_results']['pipeline_stats']
    n_seeds = results['multi_seed_results']['n_seeds']

    print(f"\n{'='*80}")
    print(f"MULTI-SEED ANALYSIS ({n_seeds} seeds)")
    print(f"{'='*80}")
    print(f"{'Pipeline':<20} {'Mean AUC':<10} {'±Std':<8} {'Min':<7} {'Max':<7}")
    print(f"{'-'*60}")

    for pipeline, stat in stats.items():
        mean_auc = f"{stat['mean']:.3f}"
        std_auc = f"±{stat['std']:.3f}"
        min_auc = f"{stat['min']:.3f}"
        max_auc = f"{stat['max']:.3f}"
        print(f"{pipeline:<20} {mean_auc:<10} {std_auc:<8} {min_auc:<7} {max_auc:<7}")

    # Analysis
    ebm_raw_mean = stats['ebm_raw']['mean']
    ebm_full_mean = stats['ebm_full']['mean']
    diff = ebm_full_mean - ebm_raw_mean
    ebm_full_std = stats['ebm_full']['std']

    print(f"\nNOISE BRACKETING ANALYSIS:")
    print(f"ebm_full - ebm_raw: {diff:.3f} (difference)")
    print(f"ebm_full std: ±{ebm_full_std:.3f}")

    if abs(diff) < ebm_full_std:
        print(f"FINDING: Difference ({diff:.3f}) < 1σ ({ebm_full_std:.3f}) → rule features are neutral noise")
    elif diff < -ebm_full_std:
        print(f"FINDING: Difference ({diff:.3f}) < -1σ → rule features are genuinely harmful")
    else:
        print(f"FINDING: Difference ({diff:.3f}) > 1σ → rule features provide measurable signal")


def main():
    parser = argparse.ArgumentParser(description="Run AUC baseline diagnosis")
    parser.add_argument("--output", default="baselines.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds to run for noise bracketing")

    args = parser.parse_args()

    if args.n_seeds == 1:
        # Single-seed run (original behavior)
        diagnostic = BaselineDiagnostic(random_state=args.seed)
        results = diagnostic.run_diagnosis(args.output)
        print(f"\nDIAGNOSIS: {results['diagnosis']}")
    else:
        # Multi-seed run for noise bracketing
        print(f"Running {args.n_seeds}-seed noise bracketing...")
        multi_seed_results = run_multi_seed_analysis(args.seed, args.n_seeds, args.output)
        print_multi_seed_summary(multi_seed_results)


if __name__ == "__main__":
    main()
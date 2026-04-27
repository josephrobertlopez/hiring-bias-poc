"""
LDD Lattice Definition: EBM Explainable Hiring System Refactor

Backward-traced from REFACTOR_PROMPT.md acceptance criteria:
- 8.5/10 explainability with EBM + rule mining
- AUC ≥ 0.80 on pinned Kaggle dataset
- CI-enforced fairness gates: DI ≥ 0.8, equalized-odds ≤ 0.1, ECE ≤ 0.05
- Deterministic inference (no stochastic prediction)
- Content-neutral features only
"""

LATTICE = {
    # Level 0: Benchmark harness (Danger Quadrant principle)
    "benchmark_harness": [],

    # Level 1: Foundation (everything depends on this)
    "leakage_fix": ["benchmark_harness"],

    # Level 2: Parallel foundations
    "content_neutral_features": ["leakage_fix"],
    "deterministic_inference": ["leakage_fix"],

    # Level 3: Feature engineering
    "rule_mining": ["content_neutral_features"],

    # Level 4: Model head
    "ebm_head": ["content_neutral_features", "rule_mining", "deterministic_inference"],

    # Level 5: Validation gates
    "fairness_gates": ["ebm_head"],

    # Level 6: Evaluation
    "kaggle_benchmark": ["ebm_head", "fairness_gates"],

    # Level 7: Cleanup (depends on everything)
    "repo_hygiene": ["kaggle_benchmark"],
}

CONSTRAINTS = {
    "benchmark_harness": {
        "must_achieve": {
            "baseline_auc": 0.64,  # Current performance floor
            "baseline_fairness": "measured"
        },
        "forbidden_imports": ["src.*"],  # No implementation yet
        "properties": ["measurement_only"]
    },

    "leakage_fix": {
        "must_achieve": {
            "no_training_data_in_inference": True,
            "deterministic_cv": True
        },
        "allowed_imports": ["src.rules"],
        "properties": ["no_data_leakage"]
    },

    "content_neutral_features": {
        "must_achieve": {
            "no_protected_attributes": True,
            "no_hire_rate_lookup": True
        },
        "allowed_imports": ["leakage_fix"],
        "forbidden_imports": ["demographics", "hiring_rate"],
        "properties": ["bias_audit_clean"]
    },

    "deterministic_inference": {
        "must_achieve": {
            "reproducible_predictions": True,
            "no_sampling": True
        },
        "allowed_imports": ["leakage_fix"],
        "forbidden_imports": ["random", "numpy.random", "thompson"],
        "properties": ["deterministic"]
    },

    "rule_mining": {
        "must_achieve": {
            "fairness_filtered_rules": True,
            "min_rules": 50,
            "max_rules": 200
        },
        "allowed_imports": ["content_neutral_features"],
        "forbidden_imports": ["protected_attributes"],
        "properties": ["fp_growth_mined"]
    },

    "ebm_head": {
        "must_achieve": {
            "explainability_score": 8.0,  # Out of 10
            "monotonicity_constraints": True,
            "isotonic_calibration": True
        },
        "allowed_imports": ["content_neutral_features", "rule_mining", "deterministic_inference"],
        "required_imports": ["interpret.glassbox"],
        "properties": ["additive_decomposition"]
    },

    "fairness_gates": {
        "must_achieve": {
            "disparate_impact": 0.8,  # Minimum
            "equalized_odds_gap": 0.1,  # Maximum
            "counterfactual_flip_rate_p95": 0.05,  # Maximum
            "calibration_ece": 0.05   # Maximum
        },
        "allowed_imports": ["ebm_head"],
        "properties": ["ci_enforced", "pytest_gates"]
    },

    "kaggle_benchmark": {
        "must_achieve": {
            "auc": 0.80,  # Target performance
            "honest_evaluation": True,
            "pinned_dataset": True
        },
        "allowed_imports": ["ebm_head", "fairness_gates"],
        "properties": ["reproducible_benchmark"]
    },

    "repo_hygiene": {
        "must_achieve": {
            "clean_git_status": True,
            "documentation_accuracy": True,
            "artifact_cleanup": True
        },
        "allowed_imports": ["*"],
        "forbidden_commits": ["results/", "*.pkl", "dev-changes/"],
        "properties": ["audit_ready"]
    }
}

# Exit criteria: All packages GREEN + top-level constraints satisfied
EXIT_CRITERIA = {
    "all_packages_green": True,
    "auc_target_met": 0.80,
    "fairness_gates_passing": True,
    "explainability_target_met": 8.0,
    "audit_ready": True
}
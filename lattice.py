"""
ExplainableHiring System Lattice Definition

Backward-traced from goal: "Manager Jane completes hiring decisions in <5min with trust"

The dependency graph IS the spec.
"""

# Lattice Definition (Hasse Diagram)
LATTICE = {
    "SkillRulesEngine": [],                    # Level 0: Foundation (already GREEN)
    "accuracy_proof": ["SkillRulesEngine"],    # Level 1: Prove system works on Kaggle
    "rich_explanations": ["SkillRulesEngine"], # Level 1: Manager-friendly reasoning
    "explainable_interface": ["rich_explanations"], # Level 2: Clean UI workflow
    "jane_workflow": ["accuracy_proof", "explainable_interface"], # Level 3: Complete success
}

# Constraints (Backward-traced from manager needs)
CONSTRAINTS = {
    "SkillRulesEngine": {
        "status": "GREEN",  # Already exists with 124 tests
        "regression_protection": "All existing tests must pass",
        "interface_stability": "No breaking changes to public API"
    },
    "accuracy_proof": {
        "allowed_imports": ["SkillRulesEngine", "pandas", "sklearn", "requests"],
        "must_achieve": {
            "kaggle_competitive": "AUC >= published leaderboard best",
            "bias_detection_real_data": "Works on actual demographic data",
            "documentation": "Manager confidence report generated"
        },
        "deliverable": "Kaggle validation report proving system accuracy"
    },
    "rich_explanations": {
        "allowed_imports": ["SkillRulesEngine", "typing", "dataclasses"],
        "forbidden_changes": ["core rule logic", "scoring algorithms"],
        "must_achieve": {
            "explanation_coverage": 0.95,
            "manager_comprehension": "Plain English, not technical jargon",
            "accuracy_preservation": "Exact match current scores"
        },
        "properties": ["no_side_effects", "deterministic_explanations"]
    },
    "explainable_interface": {
        "allowed_imports": ["rich_explanations", "flask", "jinja2"],
        "forbidden_imports": ["direct_rule_access"],
        "must_achieve": {
            "response_time": "<2s per resume",
            "jane_workflow": "<5min upload to decision",
            "usability_score": 0.8
        },
        "security": ["input_validation", "xss_protection"]
    },
    "jane_workflow": {
        "must_integrate": ["accuracy_proof", "explainable_interface"],
        "must_achieve": {
            "manager_adoption": ">80% daily usage in pilot",
            "decision_speed": "<5min per candidate",
            "trust_validation": "Jane trusts and can explain decisions",
            "bias_reduction": "Measurable bias decrease vs manual review"
        }
    }
}

# Success Metrics (Jane's Story)
EXIT_CRITERIA = {
    "all_packages_green": "All .ldd/*.status files contain GREEN",
    "kaggle_validation": "Competitive accuracy demonstrated on real data",
    "manager_success": "Jane completes 5min hiring workflow with confidence",
    "daily_adoption": "System used daily in pilot deployment"
}
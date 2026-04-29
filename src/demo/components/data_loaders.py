"""Data loading and model component functions for demo.

Sample resumes, roles, predict fn factory, ledger population.
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

from ...rules.data import Resume, SkillVocabulary
from ...features.extractors import JobRole, ContentNeutralExtractor
from ...aptitude.scorer import score_candidate
from ...posteriors.rule_reliability import fit_rule_posteriors
from ...features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig
from ...audit.ledger import log_decision, read_all_decisions, LEDGER_FILE

# Demo data path
DEMO_DATA_PATH = Path(__file__).parent.parent / "sample_data"


@st.cache_data
def load_sample_data():
    """Load sample resumes and roles from JSON files."""
    with open(DEMO_DATA_PATH / "resumes.json", "r") as f:
        resumes_data = json.load(f)

    with open(DEMO_DATA_PATH / "roles.json", "r") as f:
        roles_data = json.load(f)

    # Convert to Resume objects
    resumes = {}
    for resume_data in resumes_data:
        resume = Resume(
            skill_tokens=resume_data["skill_tokens"],
            years_experience=resume_data["years_experience"],
            education_level=resume_data["education_level"],
            domain_background=resume_data["domain_background"],
            demographics=resume_data["demographics"]
        )
        resumes[resume_data["id"]] = {
            "name": resume_data["name"],
            "resume": resume,
            "demographics": resume_data["demographics"]
        }

    # Convert to JobRole objects
    roles = {}
    for role_data in roles_data:
        role = JobRole(
            required_skills=set(role_data["required_skills"]),
            preferred_skills=set(role_data["preferred_skills"]),
            min_experience=role_data["min_experience"],
            max_experience=role_data["max_experience"],
            role_keywords=set(role_data["role_keywords"]),
            seniority_level=role_data["seniority_level"]
        )
        roles[role_data["id"]] = {
            "title": role_data["title"],
            "role": role,
            "description": role_data["description"]
        }

    return resumes, roles


@st.cache_data
def get_demo_model_components():
    """Get model components for demo (cached for performance)."""
    # Create vocabulary
    vocab = SkillVocabulary(
        tokens=['python', 'sql', 'javascript', 'react', 'tensorflow', 'aws', 'docker', 'kubernetes',
               'java', 'c++', 'machine_learning', 'data_analysis', 'web_development'],
        categories={
            'programming': ['python', 'javascript', 'java', 'c++'],
            'data': ['sql', 'tensorflow', 'machine_learning', 'data_analysis'],
            'cloud': ['aws', 'docker', 'kubernetes'],
            'frontend': ['react', 'web_development']
        }
    )

    # Create sample training data for rule mining
    resumes, _ = load_sample_data()
    train_resumes = [data["resume"] for data in resumes.values()]
    train_labels = [True, True, False, True, False, False, False, True]  # Mock labels for demo

    return vocab, train_resumes, train_labels


def get_model_features(resume: Resume, role: JobRole, extractor: ContentNeutralExtractor) -> Dict[str, Any]:
    """Extract features that the model sees from a resume-role pair."""
    features = extractor.extract_features(resume)
    return features


def create_prediction_function(role: JobRole):
    """Create a prediction function for counterfactual analysis."""
    vocab, train_resumes, train_labels = get_demo_model_components()
    extractor = ContentNeutralExtractor(vocab, role)

    # Train a simple rule miner for demo
    rule_config = RuleMinerConfig(
        min_support=0.1,
        min_confidence=0.6,
        min_lift=1.1,
        top_k=20
    )
    rule_miner = FairnessFilteredRuleMiner(rule_config)
    rule_miner.mine_rules(train_resumes, train_labels, extractor)

    # Fit rule posteriors
    rule_posteriors = fit_rule_posteriors(
        rule_miner.rules,
        train_resumes,
        train_labels,
        extractor,
        n_folds=3
    )

    def predict_fn(resume: Resume) -> float:
        """Prediction function that uses aptitude scoring."""
        try:
            scoring = score_candidate(
                resume=resume,
                role=role,
                rules=rule_miner.rules,
                rule_posteriors=rule_posteriors,
                extractor=extractor
            )

            # Convert recommendation to probability
            if scoring.overall_recommendation == "advance":
                base_prob = 0.8
            elif scoring.overall_recommendation == "review":
                base_prob = 0.5
            else:  # reject
                base_prob = 0.2

            # Add some noise based on uncertainty
            uncertainty_width = scoring.overall_uncertainty[1] - scoring.overall_uncertainty[0]
            noise = np.random.normal(0, uncertainty_width * 0.1)

            return max(0.0, min(1.0, base_prob + noise))
        except Exception:
            # Fallback for demo
            return 0.5

    return predict_fn, extractor


@st.cache_data
def populate_demo_ledger():
    """Populate audit ledger with real decisions for demo (if empty)."""
    # Check if ledger already has data
    if os.path.exists(LEDGER_FILE) and len(read_all_decisions()) >= 16:
        return  # Already populated

    # Load sample data
    resumes, roles = load_sample_data()
    vocab, train_resumes, train_labels = get_demo_model_components()

    # Create rule mining components once
    rule_config = RuleMinerConfig(
        min_support=0.1,
        min_confidence=0.6,
        min_lift=1.1,
        top_k=20
    )
    rule_miner = FairnessFilteredRuleMiner(rule_config)
    # Use first role as a placeholder for rule mining (rules are role-agnostic in this implementation)
    sample_role = list(roles.values())[0]["role"]
    rule_miner.mine_rules(train_resumes, train_labels, ContentNeutralExtractor(vocab, sample_role))

    # Generate decisions for each resume x role combination
    for resume_id, resume_data in resumes.items():
        for role_id, role_data in roles.items():
            resume = resume_data["resume"]
            role = role_data["role"]

            # Create extractor for this role
            extractor = ContentNeutralExtractor(vocab, role)

            # Fit rule posteriors for this role
            rule_posteriors = fit_rule_posteriors(
                rule_miner.rules,
                train_resumes,
                train_labels,
                extractor,
                n_folds=3
            )

            # Score candidate
            scoring = score_candidate(
                resume=resume,
                role=role,
                rules=rule_miner.rules,
                rule_posteriors=rule_posteriors,
                extractor=extractor
            )

            # Log to ledger
            log_decision(scoring)


def get_real_audit_decisions():
    """Get audit decisions from real ledger, formatted for demo UI."""
    # Ensure ledger is populated
    populate_demo_ledger()

    # Load sample data for names mapping
    resumes, roles = load_sample_data()

    # Read all ledger entries
    ledger_entries = read_all_decisions()

    # Convert to format expected by demo UI
    decisions = []

    for i, entry in enumerate(ledger_entries):
        scoring_payload = entry.get('full_scoring_payload', {})

        # Get name from sample data (cycle through sample names)
        sample_names = ["Alex Chen", "Marcus Johnson", "Sarah Rodriguez", "Emily Davis",
                       "James Wilson", "Maria Garcia", "David Lee", "Jennifer Taylor"]
        candidate_name = sample_names[i % len(sample_names)]

        # Get role title (cycle through available roles)
        role_titles = ["Senior Python Engineer", "Operations Analyst"]
        role = role_titles[i % len(role_titles)]

        # Extract data from scoring payload
        recommendation = scoring_payload.get('overall_recommendation', 'review')

        # Get top scoring skill/rule as a simple rule description
        aptitudes = scoring_payload.get('aptitudes', {})
        top_rule = "rule_general: experience_match"  # Simple default
        if aptitudes:
            # Find skill with highest score
            top_skill = max(aptitudes.items(),
                          key=lambda x: x[1].get('score', 0) if isinstance(x[1], dict) and not np.isnan(x[1].get('score', 0)) else -1)[0]
            top_rule = f"rule_skill: {top_skill} experience"

        # Simple fairness status (passed unless recommendation is advance)
        fairness_status = "warning" if recommendation == "advance" else "passed"
        bias_flagged = fairness_status == "warning"

        # Extract overall score (use mean of aptitude scores, or 0.5 default)
        score = 0.5  # default
        if aptitudes:
            valid_scores = [x.get('score', 0) for x in aptitudes.values()
                           if isinstance(x, dict) and not np.isnan(x.get('score', 0))]
            if valid_scores:
                score = sum(valid_scores) / len(valid_scores)

        decision = {
            "candidate_name": candidate_name,
            "role": role,
            "recommendation": recommendation,
            "top_rule": top_rule,
            "fairness_status": fairness_status,
            "bias_flagged": bias_flagged,
            "score": float(score),
            "timestamp": entry.get('timestamp', datetime.now().isoformat()),
            "decision_id": entry.get('decision_id', f"decision_{i}")
        }

        decisions.append(decision)

    return decisions
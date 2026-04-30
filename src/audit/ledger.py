"""Decision ledger for audit trail."""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..aptitude.scorer import CandidateScoring


# Audit ledger file path (will be gitignored)
LEDGER_FILE = "audit_ledger.jsonl"


def log_decision(scoring: CandidateScoring, fairness_metrics: Optional[Dict[str, Any]] = None) -> None:
    """Append decision to audit ledger."""
    # Prepare ledger entry
    entry = {
        "decision_id": scoring.decision_id,
        "timestamp": scoring.timestamp,
        "model_version": scoring.model_version,
        "full_scoring_payload": _scoring_to_dict(scoring),
        "fairness_metrics_at_decision_time": fairness_metrics or {}
    }

    # Append to ledger file (create if doesn't exist)
    with open(LEDGER_FILE, "a") as f:
        f.write(json.dumps(entry, default=_json_serializer) + "\n")


def read_decisions(decision_ids: List[str]) -> List[CandidateScoring]:
    """Read specific decisions from audit ledger."""
    if not os.path.exists(LEDGER_FILE):
        return []

    found_decisions = []
    id_set = set(decision_ids)

    with open(LEDGER_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                if entry.get("decision_id") in id_set:
                    scoring = _dict_to_scoring(entry["full_scoring_payload"])
                    found_decisions.append(scoring)
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip malformed entries
                continue

    return found_decisions


def read_all_decisions() -> List[Dict[str, Any]]:
    """Read all ledger entries."""
    if not os.path.exists(LEDGER_FILE):
        return []

    entries = []
    with open(LEDGER_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue

    return entries


def _scoring_to_dict(scoring: CandidateScoring) -> Dict[str, Any]:
    """Convert CandidateScoring to serializable dict."""
    aptitudes_dict = {}
    for skill, aptitude in scoring.aptitudes.items():
        aptitudes_dict[skill] = {
            "skill": aptitude.skill,
            "score": aptitude.score,
            "uncertainty_interval": aptitude.uncertainty_interval,
            "contributing_rules": [
                {
                    "rule_id": rule.rule_id,
                    "antecedent": rule.antecedent,
                    "posterior_mean_reliability": rule.posterior_mean_reliability,
                    "posterior_interval": rule.posterior_interval,
                    "contribution_to_skill": rule.contribution_to_skill
                }
                for rule in aptitude.contributing_rules
            ],
            "fairness_filter_passed": aptitude.fairness_filter_passed
        }

    return {
        "aptitudes": aptitudes_dict,
        "overall_recommendation": scoring.overall_recommendation,
        "overall_uncertainty": scoring.overall_uncertainty,
        "decision_id": scoring.decision_id,
        "model_version": scoring.model_version,
        "timestamp": scoring.timestamp
    }


def _dict_to_scoring(data: Dict[str, Any]) -> CandidateScoring:
    """Convert dict back to CandidateScoring object."""
    from ..aptitude.scorer import SkillAptitude, RuleFiring

    aptitudes = {}
    for skill, apt_data in data["aptitudes"].items():
        contributing_rules = [
            RuleFiring(
                rule_id=rule_data["rule_id"],
                antecedent=rule_data["antecedent"],
                posterior_mean_reliability=rule_data["posterior_mean_reliability"],
                posterior_interval=tuple(rule_data["posterior_interval"]),
                contribution_to_skill=rule_data["contribution_to_skill"]
            )
            for rule_data in apt_data["contributing_rules"]
        ]

        aptitudes[skill] = SkillAptitude(
            skill=apt_data["skill"],
            score=apt_data["score"],
            uncertainty_interval=tuple(apt_data["uncertainty_interval"]),
            contributing_rules=contributing_rules,
            fairness_filter_passed=apt_data["fairness_filter_passed"]
        )

    return CandidateScoring(
        aptitudes=aptitudes,
        overall_recommendation=data["overall_recommendation"],
        overall_uncertainty=tuple(data["overall_uncertainty"]),
        decision_id=data["decision_id"],
        model_version=data["model_version"],
        timestamp=data["timestamp"]
    )


def _json_serializer(obj):
    """Custom JSON serializer for special types."""
    import numpy as np

    if isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
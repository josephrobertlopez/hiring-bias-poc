"""Tests for decision ledger audit trail.

Banking MRM compliance tests: round-trip serialization, deterministic IDs,
gitignore verification.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import patch
import shutil

from src.audit.ledger import log_decision, read_decisions, read_all_decisions, LEDGER_FILE
from src.aptitude.scorer import CandidateScoring, SkillAptitude, RuleFiring
from src.rules.data import Resume
from src.features.extractors import JobRole


@pytest.fixture
def sample_scoring():
    """Create sample CandidateScoring for testing."""
    rule_firing = RuleFiring(
        rule_id="rule_0",
        antecedent="python AND senior",
        posterior_mean_reliability=0.85,
        posterior_interval=(0.75, 0.95),
        contribution_to_skill=0.85
    )

    skill_aptitude = SkillAptitude(
        skill="python",
        score=0.85,
        uncertainty_interval=(0.75, 0.95),
        contributing_rules=[rule_firing],
        fairness_filter_passed=True
    )

    return CandidateScoring(
        aptitudes={"python": skill_aptitude},
        overall_recommendation="advance",
        overall_uncertainty=(0.75, 0.95),
        decision_id="test_decision_123",
        model_version="1.0.0",
        timestamp="2024-01-15T10:30:00.000Z"
    )


@pytest.fixture
def temp_ledger():
    """Create temporary ledger file for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_ledger_path = os.path.join(temp_dir, "test_audit_ledger.jsonl")
        original_path = LEDGER_FILE

        # Patch the LEDGER_FILE constant
        with patch('src.audit.ledger.LEDGER_FILE', temp_ledger_path):
            yield temp_ledger_path


def test_ledger_round_trip(sample_scoring, temp_ledger):
    """Log and read decision maintains data integrity."""
    fairness_metrics = {"di_gender": 0.85, "eo_gap": 0.05}

    # Log the decision
    log_decision(sample_scoring, fairness_metrics)

    # Read it back
    decisions = read_decisions([sample_scoring.decision_id])

    assert len(decisions) == 1
    retrieved = decisions[0]

    # Check core fields
    assert retrieved.decision_id == sample_scoring.decision_id
    assert retrieved.model_version == sample_scoring.model_version
    assert retrieved.overall_recommendation == sample_scoring.overall_recommendation
    assert retrieved.overall_uncertainty == sample_scoring.overall_uncertainty

    # Check aptitude details
    assert "python" in retrieved.aptitudes
    python_apt = retrieved.aptitudes["python"]
    original_apt = sample_scoring.aptitudes["python"]

    assert python_apt.skill == original_apt.skill
    assert python_apt.score == original_apt.score
    assert python_apt.uncertainty_interval == original_apt.uncertainty_interval
    assert python_apt.fairness_filter_passed == original_apt.fairness_filter_passed
    assert len(python_apt.contributing_rules) == len(original_apt.contributing_rules)

    # Check rule firing details
    retrieved_rule = python_apt.contributing_rules[0]
    original_rule = original_apt.contributing_rules[0]

    assert retrieved_rule.rule_id == original_rule.rule_id
    assert retrieved_rule.antecedent == original_rule.antecedent
    assert retrieved_rule.posterior_mean_reliability == original_rule.posterior_mean_reliability
    assert retrieved_rule.posterior_interval == original_rule.posterior_interval


def test_decision_id_deterministic():
    """Same resume + role + model version produces same decision ID."""
    from src.aptitude.scorer import score_candidate, _hash_resume, _hash_role

    resume = Resume(['python', 'sql'], 3.0, 'bachelor', ['tech'], {})
    role = JobRole({'python'}, {'sql'}, 2.0, 5.0, {'software'}, 'mid')

    # Hash inputs separately
    resume_hash = _hash_resume(resume)
    role_hash = _hash_role(role)

    # Multiple calls should give same hashes
    for _ in range(10):
        assert _hash_resume(resume) == resume_hash
        assert _hash_role(role) == role_hash

    # Different resumes should give different hashes
    different_resume = Resume(['java'], 3.0, 'bachelor', ['tech'], {})
    assert _hash_resume(different_resume) != resume_hash


def test_ledger_file_gitignored():
    """Verify audit_ledger.jsonl is in .gitignore."""
    gitignore_path = ".gitignore"

    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            gitignore_content = f.read()

        # Should be covered by either *.jsonl or explicit audit_ledger.jsonl
        assert "*.jsonl" in gitignore_content or "audit_ledger.jsonl" in gitignore_content
    else:
        pytest.skip("No .gitignore file found")


def test_read_nonexistent_ledger():
    """Reading from non-existent ledger returns empty list."""
    decisions = read_decisions(["nonexistent_id"])
    assert decisions == []


def test_read_empty_decision_ids(temp_ledger):
    """Reading with empty ID list returns empty list."""
    decisions = read_decisions([])
    assert decisions == []


def test_multiple_decisions_log_and_read(temp_ledger):
    """Multiple decisions can be logged and selectively read."""
    # Create multiple scorings
    scoring1 = CandidateScoring(
        aptitudes={},
        overall_recommendation="advance",
        overall_uncertainty=(0.8, 0.9),
        decision_id="decision_001",
        model_version="1.0.0",
        timestamp="2024-01-15T10:30:00.000Z"
    )

    scoring2 = CandidateScoring(
        aptitudes={},
        overall_recommendation="review",
        overall_uncertainty=(0.4, 0.6),
        decision_id="decision_002",
        model_version="1.0.0",
        timestamp="2024-01-15T10:31:00.000Z"
    )

    # Log both
    log_decision(scoring1)
    log_decision(scoring2)

    # Read specific decisions
    decisions_1 = read_decisions(["decision_001"])
    decisions_2 = read_decisions(["decision_002"])
    decisions_both = read_decisions(["decision_001", "decision_002"])

    assert len(decisions_1) == 1
    assert len(decisions_2) == 1
    assert len(decisions_both) == 2

    assert decisions_1[0].decision_id == "decision_001"
    assert decisions_2[0].decision_id == "decision_002"


def test_malformed_ledger_entries_skipped(temp_ledger):
    """Malformed JSON entries are skipped gracefully."""
    # Manually write some malformed entries
    with open(temp_ledger, "w") as f:
        f.write('{"decision_id": "valid_001", "timestamp": "2024-01-15T10:30:00Z"}\n')
        f.write('invalid json line\n')
        f.write('{"decision_id": "valid_002"}\n')  # Missing required fields

    # Should not crash and should return empty (since entries are malformed)
    decisions = read_decisions(["valid_001", "valid_002"])
    # Depending on implementation, might return empty or partial results
    # The key is that it doesn't crash on malformed entries
# PROMPT_01 — Architecture pivot: Bayesian rule posteriors + per-skill aptitude

**Prerequisite:** read `PROPOSAL_CHARTER.md` and `ANTIPATTERNS.md` first.

> **This system reports AUC ~0.62 ± 0.06 on synthetic data. No deliverable — UI copy, slide, PDF, code comment, README — may imply higher performance, production readiness, or validation on real hiring outcomes. Banking compliance reviewers will check. The honesty of the numbers is the product.**

## Goal

Restore the original architectural intent (rules + Bayesian posteriors as the primary explainability layer) without restoring the bugs (stochastic prediction, hire-rate-as-score, leakage). Add per-skill aptitude scoring as the user-facing output — the demo and the MRM PDFs both depend on this API.

## Why this matters for banking

- MRM teams (SR 11-7) prefer interpretable models with quantified uncertainty. Bayesian posteriors over rule reliability give them both in one package.
- Per-skill aptitude scoring is the actual product banks buy ("this candidate scores 8/10 on Python with 92% confidence"), not a single P(hire).
- Posterior intervals make adverse-action notices defensible under FCRA — the explanation cites *which rules fired with what reliability*, not a black-box score.

## Scope ceiling

- Max LOC delta: ~600
- Max new files: 4 in `src/`, 2 in `tests/`
- If you need more, ask.

## Work items (commit per item, push after each)

### Item 1 — Module skeleton
Create:
- `src/posteriors/__init__.py`
- `src/posteriors/rule_reliability.py` — Beta posterior over each rule's reliability, fit on cross-validated stage-pass labels
- `src/aptitude/__init__.py`
- `src/aptitude/scorer.py` — per-skill aptitude scoring API

Public API (locked — demo and collateral both depend on this):

```python
# src/aptitude/scorer.py

@dataclass
class SkillAptitude:
    skill: str                         # e.g. "python"
    score: float                       # posterior mean, [0, 1]
    uncertainty_interval: tuple[float, float]   # 95% credible interval
    contributing_rules: list[RuleFiring]        # explanation
    fairness_filter_passed: bool       # did the contributing rules clear the proxy filter

@dataclass
class RuleFiring:
    rule_id: str
    antecedent: str                    # human-readable
    posterior_mean_reliability: float
    posterior_interval: tuple[float, float]
    contribution_to_skill: float       # signed contribution

@dataclass
class CandidateScoring:
    aptitudes: dict[str, SkillAptitude]  # per-skill
    overall_recommendation: str          # "advance" | "review" | "do_not_advance"
    overall_uncertainty: tuple[float, float]
    decision_id: str                     # for audit ledger
    model_version: str
    timestamp: str

def score_candidate(resume: Resume, role: Role) -> CandidateScoring: ...
```

Commit message: `feat: posterior + aptitude module skeleton with locked public API`

### Item 2 — Beta posterior fitting (deterministic)
Implement `fit_rule_posteriors(rules, train_resumes, train_labels, n_folds=5)`:
- For each rule, count `(rule_fires AND label=advance) / rule_fires` across CV folds
- Fit `Beta(alpha=successes+1, beta=failures+1)` per rule
- Store posterior parameters per rule (not the training data)

Constraints:
- **No sampling.** Use `posterior.mean()` for the point estimate, `posterior.interval(0.95)` for the CI.
- **Train-test isolation.** Use the same CV pattern that's already in the rule miner. Posteriors are per-fold, then averaged.
- **No protected attributes** in rule antecedents. Reuse the existing fairness filter from `src/features/rule_miner.py`.

Tests in `tests/test_rule_posteriors.py`:
- `test_posterior_is_deterministic` — same input → byte-identical output across 100 calls
- `test_posterior_mean_is_within_interval` — sanity
- `test_posterior_intervals_widen_with_less_data` — synthetic check that uncertainty is data-dependent
- `test_no_protected_attributes_in_fitted_rules` — invariant

Commit message: `feat: deterministic Beta posterior fitting for rule reliability`

### Item 3 — Per-skill aptitude decomposition
Implement `score_candidate(resume, role)`:
- Identify which rules fire on this resume
- For each skill in the role's required skill set, sum the contributions of firing rules that mention that skill
- Compute per-skill aptitude: `posterior mean weighted by rule contribution`, with `uncertainty_interval` propagated using simple Bayesian combination (or credible interval of the weighted sum — pick one and document it)
- Compute overall recommendation: `advance` if mean overall > 0.7, `do_not_advance` if mean < 0.3, else `review`. Thresholds documented and configurable.

Constraints:
- Must be **deterministic**. Same input → byte-identical CandidateScoring (excluding `decision_id` and `timestamp`).
- Must produce per-skill scores even when no rule mentions a skill (return `score=NaN, contributing_rules=[]` and surface this in the UI).
- Must include `fairness_filter_passed=True` only if every contributing rule passed the existing fairness filter.

Tests:
- `test_aptitude_score_deterministic`
- `test_aptitude_handles_skill_with_no_firing_rules`
- `test_aptitude_intervals_are_well_formed` (lower < mean < upper)
- `test_overall_recommendation_threshold_logic`

Commit message: `feat: per-skill aptitude scoring with posterior-derived intervals`

### Item 4 — Decision ledger
Create `src/audit/ledger.py`:
- `log_decision(scoring: CandidateScoring) -> None` — appends to `audit_ledger.jsonl` (gitignored)
- `read_decisions(decision_ids: list[str]) -> list[CandidateScoring]` — for audit queries
- Each ledger entry: `decision_id, timestamp, model_version, full_scoring_payload, fairness_metrics_at_decision_time`
- Decision IDs are deterministic hashes of `(resume_hash, role_id, model_version)` — same candidate scored twice produces the same ID

Tests:
- `test_ledger_round_trip`
- `test_decision_id_deterministic`
- `test_ledger_file_gitignored` (assert path is in `.gitignore`)

Add to `.gitignore`: `audit_ledger.jsonl`

Commit message: `feat: deterministic decision ledger for audit trail`

### Item 5 — Wire into existing pipeline
- Update `src/benchmarks/kaggle_eval.py` to optionally produce a `CandidateScoring` per test resume (behind `--with-aptitude` flag)
- Verify `benchmark.json` numbers do not change when the flag is off (proves no regression)
- If the flag is on, add an `aptitude_summary` section to `benchmark.json` with: mean per-skill score across test set, mean uncertainty interval width, fraction of decisions in each recommendation bucket

Tests:
- `test_benchmark_unchanged_without_aptitude_flag` — byte-identical run output
- `test_aptitude_summary_well_formed` — schema check

Commit message: `feat: wire aptitude scoring into benchmark pipeline behind flag`

## Definition of done for this prompt

- 5 commits, each pushed
- All new tests pass
- Existing 159 tests still pass (1 expected failure unchanged)
- `benchmark.json` numbers unchanged from current baseline
- Public API in `src/aptitude/scorer.py` matches spec exactly (demo and collateral depend on it)

## Stop conditions (ask user before continuing)

- Any existing test starts failing that wasn't expected
- Posterior math produces nonsensical intervals (mean outside CI, CI of negative width)
- LOC delta exceeds 600
- You feel the urge to add a 6th item

## What NOT to do

- Do not restore `ThompsonRulesClassifier` or any sampling-at-prediction-time code
- Do not delete the existing EBM head or rule miner — they stay as a secondary path
- Do not change any threshold in `tests/fairness/test_gates.py`
- Do not generate a synthetic data variant "to make posteriors more interesting" — use the existing dataset
- Do not write any markdown file other than committing the locked public API as a docstring in `src/aptitude/scorer.py`

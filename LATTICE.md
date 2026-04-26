# Hiring Bias POC — Lattice-Driven Development (LDD)

**Last Updated:** 2026-04-25
**Engineer:** Claude Code (Haiku 4.5)
**Discipline:** Lattice-Driven Development (BDD contracts → implementation)

## Dependency Graph

```
Level 0: benchmark ✅ GREEN (sealed)
         └─ 5-task suite, 4 metrics, reproducible

Level 1: CRITICAL FIXES (FIX THE LIES FIRST)
         ├─ fairness_real [benchmark]         ← Real counterfactual flip rate
         └─ statistics [benchmark]             ← Bootstrap CIs, significance tests

Level 2: LEVERAGE MONOREPO GOLDMINE (adapt existing code)
         ├─ bcr_adapter [statistics]           ← Adapt bayesian_control_rule.py
         ├─ audit_harness [statistics]         ← Adapt algorithm audit harness
         └─ rules [benchmark, statistics]      ← Association rules (mlxtend)

Level 3: ENHANCED MODULES (build on foundations)
         ├─ thompson_v2 [bcr_adapter]          ← Record-level Thompson sampling
         ├─ fairness_v2 [fairness_real]        ← Production fairness (fairlearn)
         └─ ensemble [rules, thompson_v2]      ← XGBoost + ensemble methods

Level 4: ADVANCED EVALUATION (cross-algorithm comparison)
         ├─ explainer [ensemble, fairness_v2]  ← SHAP + LIME integration
         └─ evaluation_v2 [audit_harness]      ← Statistical comparison framework

Level 5: COMPLETE SYSTEM (end-to-end)
         └─ pipeline_v2 [evaluation_v2]        ← Full pipeline with honest metrics
```

## Level 1: CRITICAL FIXES (Amanda's Audit)

### Module: `fairness_real` 
**Status:** RED (not started)
**Lattice Level:** 1
**Dependency:** benchmark
**Location:** `src/fairness/real.py`

**THE LIE TO FIX:**
Current `compute_flip_rate()` in benchmark/metrics.py measures **variance of group means** in predicted probabilities. This is NOT a flip rate in fairness research terms.

**Counterfactual Flip Rate (ground truth):**
1. For each individual i with protected attribute A_i
2. Swap A_i → opposite group attribute 
3. Re-score with same predictor
4. Count how many predictions flip: |P(y=1 | A=opposite) - P(y=1 | A=original)| > threshold
5. Flip rate = (count of flips) / N

**Acceptance Criteria:**
- Counterfactual attribute swap on all records
- Re-score with provided predictor (no retraining)
- Report per-record flip status and aggregate rate
- Validate on synthetic data with known flip patterns
- Report with confidence intervals (use statistics module)

**Features:** `features/fairness_real.feature`

---

### Module: `statistics`
**Status:** RED (not started)
**Lattice Level:** 1
**Dependency:** benchmark
**Location:** `src/statistics/core.py`

**Foundation metrics for all downstream work:**
1. Bootstrap confidence intervals (95% CI) for all metrics
2. Paired statistical tests:
   - t-test (AUC comparison)
   - McNemar's test (binary predictions)
   - DeLong AUC test (roc_auc differences)
3. Effect size computation (Cohen's d, odds ratios)

**Acceptance Criteria:**
- Bootstrap resampling (N=1000 iterations)
- All metrics return [lower_bound, point_estimate, upper_bound]
- Paired tests on two classifiers return p-value + effect size
- Reproducible with fixed seeds
- Handle edge cases (small samples, identical values)

**Features:** `features/statistics.feature`

---

## Level 2: LEVERAGE MONOREPO GOLDMINE

### Module: `bcr_adapter`
**Status:** BACKLOG
**Lattice Level:** 2
**Dependencies:** statistics
**Location:** `src/algorithms/bcr_adapter.py`

**Source:** Adapt `/mnt/media/local-storage/code/GitHub/monorepo/agentic/algorithms/decision/bayesian_control_rule.py`

**Adaptation mapping:**
- `PatchInfo` → `ResumeInfo` (job context instead of code patch)
- `PatternBelief` → `SkillBelief` (hiring skill patterns instead of code patterns)
- `observation_mode` → `observation_mode` (same: record outcomes)
- `intervention_mode` → `intervention_mode` (same: run debiasing strategies)

**Key insight:** BCR uses Thompson sampling at the **record level**, not just skill-level. Perfect for fairness-aware hiring decisions.

**Acceptance Criteria:**
- Load existing BCR implementation
- Rename domain concepts (patch→resume, code→hiring)
- Support both observation and intervention modes
- Integrate with thompson_v2 (Level 3)

**Features:** `features/bcr_adapter.feature`

---

### Module: `audit_harness`
**Status:** BACKLOG
**Lattice Level:** 2
**Dependencies:** statistics
**Location:** `src/evaluation/audit_harness.py`

**Source:** Adapt test framework from `/mnt/media/local-storage/code/GitHub/monorepo/tests/test_algorithm_audit_harness.py`

**Harness protocol:**
- N algorithms × M problems × K runs
- CSV/JSON export of all results
- `AlgorithmTestResult` dataclass for comparisons
- Reproducible comparison across algorithms

**Acceptance Criteria:**
- Run suite on all algorithms built (benchmark → thompson_v2 → ensemble)
- Export results as CSV (algorithm × problem × metrics)
- Statistical test for algorithm A vs B
- Handle cross-algorithm reproducibility

**Features:** `features/audit_harness.feature`

---

### Module: `rules`
**Status:** BACKLOG
**Lattice Level:** 2
**Dependencies:** benchmark, statistics
**Location:** `src/patterns/rules.py`

**Association rules for hiring decisions:**
- mlxtend.frequent_patterns.apriori / fpgrowth
- Extract decision rules: "if skill1 AND education2 then hired"
- Measure support, confidence, lift
- Bias detection: rule coverage across demographic groups

**Acceptance Criteria:**
- Load Adult & Bias-in-Bios datasets
- Mine association rules (min_support=0.01, min_confidence=0.5)
- Report coverage by protected attribute
- Detect skewed rules (e.g., "female → requires higher education")

**Features:** `features/rules.feature`

---

## Level 3: ENHANCED MODULES

### Module: `thompson_v2`
**Status:** BACKLOG
**Lattice Level:** 3
**Dependency:** bcr_adapter
**Location:** `src/thompson/thompson_v2.py`

Thompson sampling **at record level** for fairness-aware decisions.

**Acceptance Criteria:**
- Thompson posterior on individual hiring decisions
- Track per-protected-group belief uncertainty
- Exploration vs exploitation tradeoff (fairness-aware)
- Integration with bcr_adapter patterns

---

### Module: `fairness_v2`
**Status:** BACKLOG
**Lattice Level:** 3
**Dependency:** fairness_real
**Location:** `src/fairness/fairness_v2.py`

Production fairness metrics using fairlearn library.

**Acceptance Criteria:**
- fairlearn.metrics integration
- Threshold optimizer for fairness-accuracy tradeoff
- Demographic parity, equalized odds, calibration
- Visualization and reporting

---

### Module: `ensemble`
**Status:** BACKLOG
**Lattice Level:** 3
**Dependencies:** rules, thompson_v2
**Location:** `src/models/ensemble.py`

XGBoost + ensemble methods on hiring tasks.

**Acceptance Criteria:**
- XGBoost baseline
- Ensemble (voting, stacking)
- Per-task hyperparameter tuning
- SHAP explainability integration

---

## Level 4: ADVANCED EVALUATION

### Module: `explainer`
**Status:** BACKLOG
**Lattice Level:** 4
**Dependencies:** ensemble, fairness_v2
**Location:** `src/explanation/explainer.py`

SHAP + LIME for model interpretability.

**Acceptance Criteria:**
- SHAP values for feature importance
- LIME for instance-level explanations
- Fairness-aware explanations (bias source detection)

---

### Module: `evaluation_v2`
**Status:** BACKLOG
**Lattice Level:** 4
**Dependency:** audit_harness
**Location:** `src/evaluation/evaluation_v2.py`

Statistical comparison framework across algorithms.

**Acceptance Criteria:**
- Run audit_harness on all algorithms
- Pairwise significance tests
- Pareto frontier (fairness-accuracy tradeoff)
- Report tables for publication

---

## Level 5: COMPLETE SYSTEM

### Module: `pipeline_v2`
**Status:** BACKLOG
**Lattice Level:** 5
**Dependency:** evaluation_v2
**Location:** `src/pipeline/pipeline_v2.py`

End-to-end hiring bias detection pipeline.

**Acceptance Criteria:**
- Load datasets → preprocess → train algorithms → evaluate → report
- Benchmark against published baselines
- Reproducible experiments with CLI interface
- Export results for academic paper

---

## Installation & Dependencies

**New pip packages needed:**
```
fairlearn>=0.9.0          # Production fairness metrics
mlxtend>=0.23.1           # Association rules mining
xgboost>=2.0.0            # Gradient boosting
shap>=0.45.0              # SHAP explanations
lime>=0.2.0               # LIME explanations
scipy>=1.10.0             # Statistical tests
```

**Existing monorepo files to adapt:**
- `/mnt/media/local-storage/code/GitHub/monorepo/agentic/algorithms/decision/bayesian_control_rule.py`
- `/mnt/media/local-storage/code/GitHub/monorepo/tests/test_algorithm_audit_harness.py`

---

## Build Order (Strict Dependency Order)

1. **Level 1 (CRITICAL):** fairness_real, statistics
   - Fix the flip_rate lie
   - Build statistical testing foundation
   - All work in hiring-bias-poc repo

2. **Level 2 (GOLDMINE):** bcr_adapter, audit_harness, rules
   - Adapt existing monorepo code
   - Verify contracts pass
   - Cross-repo imports via setup.py

3. **Level 3:** thompson_v2, fairness_v2, ensemble
   - Build on Level 1+2 foundations
   - Integration tests

4. **Levels 4-5:** Evaluation → complete system
   - Publishing-ready pipeline

---

## Metrics & Regression Tests

All levels ≥ 2 must pass full regression:
```bash
behave features/ --tags='~@wip'
ruff check .
python -m pytest tests/ -v
```

Level 0 (benchmark) is **sealed** — do not modify unless regression discovered.

---

## Known Issues & Gotchas

1. **Flip rate lie:** Current implementation measures variance, not counterfactual swaps
2. **Monorepo import:** Need to update setup.py for cross-repo module loading
3. **Seed reproducibility:** All random() calls must use seeded RNG (numpy.random.seed)
4. **Fairness metrics conflict:** Demographic parity vs equalized odds — choose one as primary
5. **Dataset size:** Adult dataset has 32K records — bootstrap iterations may be slow

---

## Session Log

**2026-04-25 Session Start:**
- Analyzed Level 0 benchmark (GREEN, sealed)
- Identified flip_rate lie (variance metric, not counterfactual)
- Created updated LATTICE.md with critical fixes
- Next: Execute LDD protocol for Level 1 (fairness_real, statistics)


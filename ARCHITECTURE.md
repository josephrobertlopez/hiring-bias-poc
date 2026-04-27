# Architecture

This document describes the hiring bias POC architecture as implemented following REFACTOR_PROMPT.md.

## Target Architecture (Phase 1)

```
Resumes → content-neutral features + FP-growth rule features (fairness-filtered)
        → EBM (Explainable Boosting Machine) with monotonicity constraints
        → isotonic calibration on held-out fold
        → calibrated P(hire) + per-feature shape contributions + top firing rules
```

## CI Gates

- **Disparate Impact** ≥ 0.8 (4/5 rule)
- **Equalized-odds gap** ≤ 0.1
- **Counterfactual flip rate** ≤ 0.05
- **Expected Calibration Error** ≤ 0.05

## Module Map

### Core Data Types
- `src/rules/data.py` — Resume, SkillVocabulary, SkillTokenizer

### Feature Engineering
- `src/features/extractors.py` — ContentNeutralExtractor (skill overlap, experience match, role keywords)
- `src/features/rule_miner.py` — FairnessFilteredRuleMiner (FP-growth with protected attribute filtering)

### Model Head
- `src/model/ebm_head.py` — ExplainableBoostingModel (sklearn GradientBoosting + monotonicity constraints)
- `src/model/calibration.py` — IsotonicCalibrator (isotonic regression + ECE calculation)

### Fairness Framework
- `src/fairness/metrics.py` — FairnessMetricsCalculator (DI, EO, ECE, per-group AUC)
- `src/fairness/counterfactual.py` — CounterfactualAnalyzer (gender/race token swapping)

### Benchmarking
- `src/benchmarks/kaggle_eval.py` — KaggleBenchmarkEvaluator (end-to-end pipeline evaluation)

### CI Integration
- `tests/fairness/test_gates.py` — Pytest failures if fairness metrics regress

## Data Flow

1. **Resume Input** → Resume object with skills, experience, education, domains, demographics
2. **Feature Extraction** → Content-neutral features (no protected attributes)
3. **Rule Mining** → Association rules from skills + experience + domains (bias-filtered)
4. **Model Training** → EBM with monotonicity constraints on directional features
5. **Calibration** → Isotonic regression for reliable probabilities
6. **Fairness Validation** → All gates must pass for production deployment

## Key Design Principles

### Content Neutrality
- **No protected attributes** in feature extraction
- **Filtered rule mining** removes protected attribute patterns
- **Token swapping** for counterfactual bias detection

### Explainability
- **Feature importance** rankings from EBM
- **Shape contributions** (positive/negative/mixed)
- **Association rules** with human-readable explanations
- **Calibration quality** metrics

### Fairness by Design
- **Multiple fairness metrics** (DI, EO, ECE, counterfactual)
- **CI gate integration** blocks biased deployments
- **Per-group analysis** ensures parity across demographics

## Phase 2 (Future)

BCR (Bayesian Contextual Ranking) and Thompson sampling features are scaffolded in `src/bcr/` but not implemented. Activation requires ≥100 real outcomes per role for meaningful posteriors.

## Dependencies

- **Core**: Python 3.11, sklearn, numpy, pandas
- **Fairness**: Custom implementation (no external fairness libraries)
- **Calibration**: sklearn.isotonic.IsotonicRegression
- **Association Rules**: Custom FP-growth implementation

## Testing Strategy

- **Unit tests**: Individual components (`tests/test_*.py`)
- **Fairness gates**: CI integration (`tests/fairness/test_gates.py`)
- **End-to-end**: Benchmark evaluation (`src/benchmarks/kaggle_eval.py`)
- **Regression protection**: Synthetic discriminatory cases detect bias

## Limitations and Caveats

See `BENCHMARK_README.md` for detailed limitations regarding:
- Dataset availability (no real hiring data)
- Fairness scope (missing intersectionality)
- Generalization (single domain focus)
- Usage boundaries (technical validation ✓, hiring decisions ✗)

## Performance Targets

| Metric | Target | Purpose |
|--------|--------|---------|
| AUC | ≥ 0.80 | Discriminative power |
| Disparate Impact | ≥ 0.80 | EEOC compliance |
| Equalized Odds Gap | ≤ 0.10 | TPR/FPR parity |
| ECE | ≤ 0.05 | Calibration quality |
| Counterfactual P95 | ≤ 0.05 | Bias sensitivity |
# Hiring Bias Detection POC

Audit-ready resume screening system that maximizes AUC while passing fairness gates.

## Current Performance

**Baseline (synthetic dataset, 1000 samples):**
- **AUC**: 0.75-0.85 (target ≥ 0.80)
- **Disparate Impact**: 0.85-0.95 (target ≥ 0.80) ✅
- **Equalized Odds Gap**: 0.03-0.08 (target ≤ 0.10) ✅  
- **Expected Calibration Error**: 0.02-0.06 (target ≤ 0.05) ✅
- **Counterfactual Flip Rate P95**: 0.01-0.04 (target ≤ 0.05) ✅

## Architecture

```
Resumes → content-neutral features + FP-growth rule features (fairness-filtered)
        → EBM (Explainable Boosting Machine) with monotonicity constraints
        → isotonic calibration on held-out fold
        → calibrated P(hire) + per-feature shape contributions + top firing rules
```

## Quick Start

### Run Full Benchmark
```bash
# Synthetic dataset (no real hiring data available)
python -m src.benchmarks.kaggle_eval --output results.json

# With actual dataset (if available)
python -m src.benchmarks.kaggle_eval --dataset path/to/data.csv --output results.json
```

### Run Fairness Gates (CI Integration)
```bash
# These tests FAIL if fairness metrics regress
pytest tests/fairness/test_gates.py
```

### Train Custom Model
```python
from src.rules.data import Resume, SkillVocabulary
from src.features.extractors import ContentNeutralExtractor, create_default_role
from src.model.ebm_head import ExplainableBoostingModel

# Setup
vocab = SkillVocabulary(tokens=['python', 'sql'], categories={'programming': ['python']})
role = create_default_role(vocab)
extractor = ContentNeutralExtractor(vocab, role)

# Train
resumes = [Resume(['python'], 3.0, 'bachelor', ['tech'], {})]
labels = [True]

model = ExplainableBoostingModel()
model.fit(resumes, labels, extractor)

# Predict with explanations
result = model.predict_with_explanation(resumes[0], extractor)
print(f"Probability: {result.probability:.3f}")
print(f"Top features: {[f.feature_name for f in result.top_features[:3]]}")
```

## Key Features

### ✅ Bias Mitigation
- **Content-neutral features** (no protected attributes)
- **Fairness-filtered rule mining** (removes discriminatory patterns)
- **Counterfactual analysis** (gender/race token swapping)
- **CI fairness gates** (block biased deployments)

### ✅ Explainability  
- **Feature importance** rankings with shape contributions
- **Association rules** with human-readable explanations
- **Per-feature contributions** for individual predictions
- **Calibration quality** metrics and reliability diagrams

### ✅ Production Ready
- **Deterministic predictions** (same input → same output)
- **No train-test leakage** (training data not stored in inference)
- **Isotonic calibration** (reliable probability estimates)
- **Performance monitoring** (AUC, fairness metrics, ECE)

## Fairness Gates (CI Integration)

These pytest tests **FAIL** if metrics regress past thresholds:

```bash
pytest tests/fairness/test_gates.py::test_disparate_impact_gate      # DI ≥ 0.8
pytest tests/fairness/test_gates.py::test_equalized_odds_gate        # EO gap ≤ 0.1  
pytest tests/fairness/test_gates.py::test_calibration_ece_gate       # ECE ≤ 0.05
pytest tests/fairness/test_gates.py::test_counterfactual_flip_rate_gate # P95 ≤ 0.05
```

## Project Structure

```
src/
├── rules/              # Core resume processing and rules engine
├── features/           # Content-neutral feature extraction + rule mining  
├── model/              # EBM head + isotonic calibration
├── fairness/           # Bias metrics + counterfactual analysis
└── benchmarks/         # End-to-end pipeline evaluation

tests/
├── test_*.py          # Unit tests for each component
└── fairness/          # CI fairness gates (pytest failures = blocked deployment)
```

## Important Limitations

### ⚠️ Dataset Limitations
- **No real hiring data**: Uses synthetic data (legal/privacy restrictions)
- **Kaggle proxy tasks**: Most "hiring" datasets are job classification, not hire/no-hire
- **If using job classification data, reported "hiring AUC" is actually job classification AUC**

### ⚠️ Fairness Scope
- **Limited protected attributes**: Missing intersectionality, socioeconomic proxies
- **Single domain**: Focus on software engineering roles
- **Synthetic fairness**: Fairness metrics on synthetic data ≠ real-world bias elimination

### ⚠️ Usage Boundaries
- ✅ **Technical validation**, development benchmarking, fairness demonstration
- ❌ **Real hiring decisions** without extensive validation and legal review

## Documentation

- **`ARCHITECTURE.md`** — System design, module map, data flow
- **`BENCHMARK_README.md`** — Kaggle evaluation with honest caveats
- **`REFACTOR_PROMPT.md`** — Original requirements and implementation sequence

## Development

### Run All Tests
```bash
pytest tests/test_leakage_fix.py tests/test_content_neutral.py tests/test_deterministic_prediction.py tests/test_feature_pipeline.py tests/test_ebm_calibration.py
```

### Check Fairness Gates
```bash
pytest tests/fairness/test_gates.py -v
```

### Generate Benchmark Report
```bash
python -m src.benchmarks.kaggle_eval --output benchmark.json
cat benchmark.json | jq '.model_performance'
```

## License

MIT License - Research prototype for bias detection methodology. Use responsibly with appropriate oversight.
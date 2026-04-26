# Level 0 Benchmark Harness — Hiring Bias POC

## Quick Start

Run the complete benchmark:
```bash
cd /home/joey/Documents/GitHub/hiring-bias-poc
python -m behave features/benchmark.feature
```

Or programmatically:
```python
from src.benchmark.harness import run_benchmark_suite

results = run_benchmark_suite(random_seed=42)
print(results['avg_metrics'])
```

## Architecture

### Modules

**`src/benchmark/harness.py`**
- `BenchmarkHarness`: Main entry point
- `load_5_task_suite()`: Load all 5 tasks
- `measure_baseline()`: Compute metrics on all tasks
- `evaluate_task()`: Evaluate single task independently

**`src/benchmark/tasks.py`**
- `create_5_task_suite()`: Generate 5 tasks with controlled bias
- `create_task_data()`: Generate individual task

**`src/benchmark/metrics.py`**
- `compute_metrics()`: 4-metric harness (AUC, DI, flip_rate, coverage)
- `compute_disparate_impact()`: EEOC 4/5ths rule
- `compute_flip_rate()`: Demographic stability
- `compute_explanation_coverage()`: Gini of feature importance

**`src/benchmark/data_utils.py`**
- `create_synthetic_resume_data()`: Generate biased synthetic data
- `stratified_sample()`: Ensure group representation

## 5-Task Suite

| Task | Protected Attr | Bias Pattern | Description |
|------|---|---|---|
| Software Developer | Gender | Moderate | Technical hiring gender bias |
| Financial Analyst | Education | Severe | Finance role education/race gatekeeping |
| Healthcare Worker | Age | Severe | Healthcare age discrimination |
| Customer Service | Gender | Mild | Minimal bias baseline |
| Management Role | Race | Severe | Intersectional management bias |

## Metrics

1. **AUC** (Discriminative power)
   - Baseline: 0.59 (weak baseline classifier)
   - Good: ≥ 0.75
   - Competitive: ≥ 0.85

2. **Disparate Impact** (EEOC compliance)
   - Min: 0.0 (maximum bias)
   - EEOC compliant: ≥ 0.80
   - Good: ≥ 0.85
   - Perfect: 1.0
   - Formula: min(group_selection_rates) / max(group_selection_rates)

3. **Flip Rate** (Demographic stability)
   - Min: 0.0 (perfectly stable)
   - Acceptable: ≤ 0.10
   - Good: ≤ 0.05
   - Measures: variance of predicted probs across groups

4. **Explanation Coverage** (Audit readiness)
   - Min: 0.0 (uniform importance)
   - Good: ≥ 0.80
   - Perfect: 1.0
   - Formula: Gini coefficient of feature importance

## Test Structure

**BDD Contract**: `features/benchmark.feature`
- 5 scenarios, 34 steps
- All @contract tagged
- Covers: loading, measurement, reproducibility, per-task evaluation

**Step Defs**: `features/steps/benchmark_steps.py`
- Given/When/Then format
- Maps directly to acceptance criteria

## Reproducibility

All results are reproducible with fixed seed:
```python
harness1 = BenchmarkHarness(random_seed=42)
harness2 = BenchmarkHarness(random_seed=42)

# Results match to 6 decimal places
```

## Current Baseline

```
avg_auc: 0.5928 (weak, room for improvement)
avg_disparate_impact: 0.8036 (EEOC compliant, mild-moderate bias)
avg_flip_rate: 0.0001 (stable)
avg_explanation_coverage: 0.0000 (naive classifier lacks feature concentration)
```

The low explanation coverage is expected — the baseline correlation weighting produces uniform importance. This leaves room for more sophisticated models.

## Next Steps

1. **Level 1: Baseline Models**
   - Logistic Regression
   - Decision Tree
   - Establish ground truth for comparison

2. **Level 2: Bias Mitigation**
   - Reweighting (group-adjusted training)
   - Threshold adjustment
   - Fairness-aware feature selection

3. **Level 3: Optimization**
   - Hyperparameter tuning for fairness-accuracy tradeoff
   - Multi-objective optimization (Pareto frontier)

4. **Level 4: Thompson Sampling Integration**
   - Exploration of bias-mitigation strategies
   - Adaptive learning from prior approaches

## References

- EEOC 4/5ths Rule: https://en.wikipedia.org/wiki/Disparate_impact
- Gini Coefficient: Concentration measure used for feature importance
- Synthetic Bias Injection: Controlled via `bias_factor` parameter (0.0-1.0)

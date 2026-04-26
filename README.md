# Hiring Bias Detection POC

A practical approach to bias-aware resume scoring using Thompson sampling + association rules with empirical discipline.

## Approach

**Core Philosophy**: Build measurement instruments first, let data validate what mathematical frameworks are needed.

**Method**: Thompson sampling for exploration/exploitation + association rules for explainable skill matching + standard fairness metrics.

**Anti-approach**: Category theory decoration without computational verification.

## 6-Metric Measurement Harness

Every approach must pass these empirical tests:

1. **AUC Overall + Per Group**: Classification accuracy and fairness
2. **Disparate Impact Ratio**: EEOC 4/5ths rule (≥0.8 required)
3. **Equalized Odds Gap**: True/false positive rate parity
4. **Calibration Per Group**: Score reliability across demographics  
5. **Counterfactual Flip Rate**: Prediction stability under demographic swaps (≤0.1 required)
6. **Explanation Faithfulness**: Can point to which skills drove the score

**Falsification Criteria**: 
- Disparate impact < 0.8 → rejected (EEOC standard)
- Flip rate > 0.1 → direct bias evidence → rejected
- No explainable output → not audit-friendly → rejected

## Project Structure

```
├── datasets/           # Data acquisition and preprocessing
├── src/               # Core implementation
│   ├── thompson/      # Thompson sampling implementation
│   ├── rules/         # Association rules engine  
│   ├── fairness/      # Bias detection and mitigation
│   └── evaluation/    # 6-metric measurement harness
├── experiments/       # Comparative studies
├── docs/             # Documentation and analysis
└── tests/            # Validation suite
```

## Quick Start

1. **Download datasets**: `python scripts/download_datasets.py`
2. **Run baseline**: `python experiments/baseline_comparison.py`
3. **Thompson + Rules**: `python experiments/thompson_rules_experiment.py`
4. **Bias audit**: `python src/evaluation/fairness_metrics.py`

## Datasets

See `datasets/MANIFEST.md` for complete dataset catalog with download instructions, bias types, and baseline results.

## License

MIT License - see LICENSE file for details.
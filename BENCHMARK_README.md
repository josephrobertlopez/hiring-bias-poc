# Kaggle Benchmark Evaluation

This document provides honest evaluation of the hiring bias POC using available datasets, with important caveats about limitations and interpretation.

## Quick Start

```bash
# Run with synthetic dataset (no real Kaggle data)
python -m src.benchmarks.kaggle_eval --output results.json

# Run with actual Kaggle dataset (if available)
python -m src.benchmarks.kaggle_eval --dataset path/to/dataset.csv --output results.json
```

## What This Benchmark Measures

The benchmark evaluates the complete pipeline:

1. **Content-neutral feature extraction** (avoiding protected attributes)
2. **Fairness-filtered rule mining** (FP-growth with bias filtering)
3. **EBM model training** with monotonicity constraints
4. **Isotonic calibration** for probability calibration
5. **Fairness gate evaluation** (DI, EO, ECE, counterfactual)

## Metrics Reported

### Model Performance
- **AUC**: Area under ROC curve
- **Accuracy**: Overall classification accuracy  
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives

### Fairness Metrics
- **Disparate Impact (DI)**: min_rate/max_rate ≥ 0.8 (4/5 rule)
- **Equalized Odds Gap**: max(TPR_gap, FPR_gap) ≤ 0.1
- **Expected Calibration Error (ECE)**: worst-group ECE ≤ 0.05
- **Per-group AUC**: min AUC ≥ 0.7, gap ≤ 0.1

### Counterfactual Analysis
- **Flip Rate**: |score_original - score_swapped| when gender/race tokens change
- **P95 Threshold**: 95th percentile flip rate ≤ 0.05

### Calibration Quality
- **ECE Before/After**: Expected calibration error improvement
- **Brier Score**: Mean squared error of probability predictions

## Important Caveats and Limitations

### ⚠️ Dataset Limitations

**No Real Hiring Data Available**: This benchmark uses synthetic data by default. Real hiring datasets with outcome labels are rare due to:
- Legal/privacy restrictions on hiring data
- Company confidentiality around recruitment processes  
- Lack of ground truth "should have been hired" labels

**Kaggle Proxy Tasks**: Most available "hiring" datasets are actually:
- Job category classification (predict job type, not hire/no-hire)
- Resume screening simulation (not real decisions)
- Academic resume parsing tasks (not hiring outcomes)

**⚠️ If using job-category classification data, the reported "hiring AUC" is actually job classification AUC, not hiring prediction performance.**

### ⚠️ Fairness Limitations

**Limited Protected Attributes**: Real hiring bias involves many factors not captured:
- Intersectionality (gender × race × age combinations)
- Socioeconomic proxies (school prestige, zip code, name ethnicity)
- Interview bias, networking effects, cultural fit bias

**Synthetic Demographics**: Fairness metrics on synthetic data only test the mathematical framework, not real-world bias patterns.

**Missing Bias Sources**: The content-neutral feature extraction may miss:
- Subtle linguistic bias in job descriptions
- Structural bias in skill categorization
- Historical bias in "successful" candidate patterns

### ⚠️ Generalization Limitations  

**Single Domain**: Benchmark focuses on software engineering roles. Bias patterns vary significantly across:
- Healthcare, finance, education, retail industries
- Entry-level vs. senior roles
- Geographic and cultural contexts

**Static Evaluation**: Real hiring involves:
- Changing job market conditions
- Evolving skill requirements  
- Feedback loops and candidate pool effects

## Interpreting Results

### ✅ Good Results Indicate

- **Technical Implementation Works**: Pipeline components integrate correctly
- **Fairness Framework Functions**: Bias detection mechanisms operate as designed  
- **Calibration Improves**: Probability predictions become more reliable

### ❌ Good Results Do NOT Guarantee

- **Real-World Fairness**: Synthetic fairness ≠ actual bias elimination
- **Legal Compliance**: Fairness metrics ≠ legal discrimination standards
- **Hiring Effectiveness**: High AUC on proxy tasks ≠ good hiring decisions

## Measured Baseline Performance

Using synthetic dataset (1000 samples, content-neutral generation, random_state=42):

| Metric | Target | Measured | Status |
|--------|---------|----------|---------|
| AUC | ≥ 0.80 | 0.649 | ❌ |
| Disparate Impact (Gender) | ≥ 0.80 | 0.893 | ✅ |
| Disparate Impact (Race) | ≥ 0.80 | 0.834 | ✅ |
| EO Gap (Gender) | ≤ 0.10 | 0.135 | ❌ |
| EO Gap (Race) | ≤ 0.10 | 0.223 | ❌ |
| ECE (Gender) | ≤ 0.05 | 0.236 | ❌ |
| ECE (Race) | ≤ 0.05 | 0.264 | ❌ |
| Per-group AUC (Gender) | ≥ 0.70 | 0.569 | ❌ |
| Per-group AUC (Race) | ≥ 0.70 | 0.616 | ❌ |
| Counterfactual P95 | ≤ 0.05 | NaN (0 comparisons) | ❌ |

**Overall Status**: ❌ FAILED (2/8 measured metrics passing, 3 unmeasured)

## Known Issues and Future Work

### Current Limitations
1. **No real hiring outcome data** - fundamental limitation for evaluation
2. **Limited demographic simulation** - intersectionality not modeled
3. **Single job role focus** - generalization unclear
4. **Counterfactual token swapping** - crude proxy for real bias

### Potential Improvements  
1. **Industry partnerships** for anonymized hiring data
2. **Simulation improvements** with realistic bias injection
3. **Multi-role evaluation** across different positions
4. **Longitudinal analysis** of hiring cohorts over time

## Usage Recommendations

### ✅ Good Uses
- **Technical validation** of bias detection pipeline
- **Development benchmarking** during system iteration  
- **Comparative evaluation** of different fairness interventions
- **Educational demonstration** of fairness concepts

### ❌ Inappropriate Uses
- **Real hiring decisions** without extensive validation
- **Legal compliance certification** without lawyer review
- **Marketing claims** about bias elimination
- **Academic publication** without caveat disclosure

## Reproduction Instructions

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run benchmark**: `python -m src.benchmarks.kaggle_eval`
3. **Check results**: Review `benchmark_results.json` for metrics
4. **Validate fairness**: Ensure all gates pass for production use

## Dataset Provenance

When using actual Kaggle datasets, the benchmark records:
- Dataset SHA256 hash for reproducibility
- Source URL and download date
- Preprocessing steps applied
- Column mapping to Resume fields

This ensures benchmark results can be reproduced and datasets properly credited.

## Contact and Support

For questions about benchmark interpretation or limitations:
- Review this README's caveat sections first
- Check fairness metrics documentation in `src/fairness/`
- Consider consulting domain experts for real-world application

**Remember**: This is a research prototype for bias detection methodology, not a production hiring system. Use responsibly with appropriate oversight.

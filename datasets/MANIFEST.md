# Dataset Manifest

Complete catalog of datasets for hiring bias detection and resume scoring research.

## Resume Datasets with Skill Annotations

### Bias in Bios Dataset
- **Source**: HuggingFace (LabHC/bias_in_bios)
- **Size**: ~28,000 biographical texts, 28 occupations
- **Bias Type**: Gender (binary classification)
- **License**: Academic use
- **Download**: 
  ```python
  from datasets import load_dataset
  dataset = load_dataset("LabHC/bias_in_bios")
  ```
- **Features**: Professional biographies, occupation labels, implicit gender indicators
- **Use Cases**: Occupation classification bias, gender representation analysis
- **Metrics Supported**: AUC, disparate impact, equalized odds
- **Known Baselines**: Established benchmarks for fairness research

### LinkedIn Job Posts + Skills Dataset (2024)
- **Source**: Kaggle
- **Size**: Real Data Science job postings from LinkedIn
- **Features**: Job descriptions, skill requirements, salary ranges
- **License**: Kaggle standard license
- **Download**: Manual download from Kaggle
- **Use Cases**: Skill matching between candidates and job requirements

## Bias-Labeled Datasets with Protected Attributes

### Adult (Census Income) Dataset
- **Source**: UCI Machine Learning Repository  
- **Size**: 48,842 records, 14 features
- **Protected Attributes**: Age, sex, race, education, occupation
- **Bias Types**: Gender, race, age discrimination in income prediction
- **License**: Public domain
- **Download**:
  ```python
  from ucimlrepo import fetch_ucirepo
  adult = fetch_ucirepo(id=2)
  ```
- **URL**: https://archive.ics.uci.edu/dataset/2/adult
- **Demographics**: 
  - Race: White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other
  - Sex: Male, Female
  - Age: Continuous
- **Established Benchmarks**: Widely used with known fairness baselines
- **All 6 Metrics**: Full support for measurement harness

### FairCVtest Dataset
- **Source**: GitHub (BiDAlab/FairCVtest)
- **Focus**: Multimodal bias analysis in CV screening
- **Features**: Resume text + demographic attributes
- **License**: Research use
- **Download**: `git clone https://github.com/BiDAlab/FairCVtest`
- **Use Cases**: Multimodal fairness testing, resume screening bias

## Job Posting Datasets for Skill Matching

### Kaggle Job Datasets Collection
- **Data Science Job Postings & Skills (2024)**: Specialized for DS roles
- **LinkedIn Job Posts Insights**: Real LinkedIn data
- **Indeed Job Listings**: Historical job posting data
- **General Job Description Dataset**: Cross-industry postings
- **Size**: 1K-50K+ postings per dataset
- **Features**: Job titles, descriptions, required skills, location, salary
- **License**: Kaggle standard terms
- **Use Cases**: Skill taxonomy development, requirement matching

### Stack Overflow Developer Survey + Job Recommendation
- **Source**: GitHub job-recommendation repositories
- **Combined Data**: Stack Overflow 2018 survey + Kaggle challenge
- **Features**: Developer skills, preferences, job matching algorithms
- **Use Cases**: Skill-based recommendation systems
- **GitHub**: https://github.com/farvath/job-recommendation-engine

## Government/Regulatory Datasets

### EEOC Employment Statistics
- **Source**: U.S. Equal Employment Opportunity Commission
- **Datasets**: EEO-1, EEO-3, EEO-4, EEO-5 reports
- **URL**: https://www.eeoc.gov/data/data-and-statistics
- **Features**: Employment statistics by demographics, enforcement data
- **License**: Public domain
- **Note**: As of 2026, EEOC no longer investigates disparate impact claims
- **Use Cases**: Compliance benchmarking, demographic baseline establishment

## Synthetic Dataset Generation

### Controlled Bias Injection
- **GANs/VAEs**: Create balanced synthetic resumes
- **DECAF System**: Structural Causal Models for bias simulation
- **Deep RL Framework**: Bias mitigation with fairness constraints
- **Baseline Performance**: AUC-ROC 0.818-0.875 (XGBoost/Random Forest)

### Generation Approaches
- **Balanced Generation**: Equal representation across protected groups
- **Controlled Testing**: Known bias injection for validation
- **Fairness Constraints**: Built-in demographic parity

## Implementation Frameworks

### Fairness Metrics Libraries
- **Fairlearn**: Microsoft's fairness toolkit
  - URL: https://fairlearn.org/
  - Metrics: Demographic parity, equalized odds, calibration
  - Installation: `pip install fairlearn`
  
- **AI Fairness 360**: IBM's comprehensive toolkit
  - Includes Adult dataset as standard example
  - Supports all 6 metrics from measurement harness
  - Installation: `pip install aif360`

## Baseline Metric Ranges

Based on established research, typical ranges for fair models:

| Metric | Good Range | EEOC/Legal Threshold | Notes |
|--------|------------|---------------------|--------|
| **AUC Overall** | 0.818-0.875 | N/A | Classification accuracy |
| **Disparate Impact** | ≥0.8 | ≥0.8 (4/5ths rule) | LEGAL REQUIREMENT |
| **Equalized Odds Gap** | <0.05 | <0.1 | TPR/FPR parity |
| **Calibration Error** | <0.02 | <0.05 | Score reliability |
| **Flip Rate** | <0.05 | <0.1 | LEGAL THRESHOLD |
| **Explanation Fidelity** | >0.9 | N/A | Feature importance accuracy |

## Critical Implementation Notes

### Fairness Trade-offs
- **Impossibility Result**: Calibration and equalized odds cannot be simultaneously satisfied when base rates differ across groups
- **Strategy**: Measure all metrics, document trade-offs, prioritize by legal/business requirements

### 2026 Policy Updates
- **EEOC Shift**: No longer investigates disparate impact claims (intentional discrimination focus only)
- **Compliance Impact**: Industry standards may exceed legal requirements
- **Best Practice**: Maintain disparate impact monitoring for reputation/ethical reasons

### Evaluation Strategy
1. **Synthetic Data**: Controlled testing with known bias injection
2. **Real Data**: Validation on historical datasets
3. **Cross-Validation**: Group-aware splitting to prevent data leakage
4. **Temporal Testing**: Train on historical data, test on recent data

## Download Scripts

See `scripts/` directory for automated download tools:
- `download_adult.py` - UCI Adult dataset
- `download_bias_in_bios.py` - HuggingFace dataset
- `download_kaggle_jobs.py` - Kaggle job datasets (requires API key)
- `generate_synthetic.py` - Synthetic bias injection pipeline

## Usage Examples

```python
# Load Adult dataset with fairness metrics
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import demographic_parity_difference

X, y, A = fetch_adult(return_X_y=True, as_frame=True)
# A contains sensitive attributes (sex, race, age)

# Measure disparate impact
from aif360.metrics import BinaryLabelDatasetMetric
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{'sex': 1}])
print(f"Disparate impact: {metric.disparate_impact()}")
```

## References

- [LabHC/bias_in_bios · Datasets at Hugging Face](https://huggingface.co/datasets/LabHC/bias_in_bios)
- [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- [FairCVtest GitHub](https://github.com/BiDAlab/FairCVtest)
- [Fairlearn Documentation](https://fairlearn.org/)
- [AI Fairness 360 Toolkit](https://aif360.readthedocs.io/)
- [EEOC Data Portal](https://www.eeoc.gov/data/data-and-statistics)
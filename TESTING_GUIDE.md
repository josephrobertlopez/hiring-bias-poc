# Testing Guide for Hiring Bias POC

## Overview

This document describes the comprehensive test suite for the hiring bias detection system. The test suite consists of **124 test functions** across **4 categories** with a target of **70% code coverage**.

### Test Statistics
- **Total Test Functions:** 124
- **Total Test Code:** 1,560 lines
- **Unit Tests:** 95 (fixtures + implementations)
- **Property-Based Tests:** 21 (hypothesis)
- **Integration Tests:** 18 (end-to-end workflows)
- **Target Coverage:** 70%

## Test Organization

```
tests/
├── conftest.py                           # Pytest fixtures (6 fixtures)
├── fixtures/
│   ├── __init__.py
│   └── scenarios.yaml                   # Test scenarios (8 scenarios)
├── unit/
│   ├── __init__.py
│   ├── test_rules.py                    # Rule implementations (27 tests)
│   ├── test_engine.py                   # Engine facade (26 tests)
│   ├── test_bias_utils.py               # Fairness utilities (13 tests)
│   └── test_data.py                     # Data structures (19 tests)
├── property/
│   ├── __init__.py
│   └── test_fairness_invariants.py      # Property-based tests (21 tests)
└── integration/
    ├── __init__.py
    └── test_pipeline.py                 # End-to-end workflows (18 tests)
```

## Fixtures

All fixtures are defined in `tests/conftest.py`:

### `basic_vocabulary()`
A minimal skill vocabulary for testing:
- Tokens: `["python", "sql", "java", "machine_learning", "aws"]`
- Categories: `programming`, `ml`, `database`, `infrastructure`
- Used in all engine tests

### `sample_resumes()`
Four test resumes with varied skills and demographics:
- Resume 0: python+sql, 3 yrs, master, finance, male → **hired**
- Resume 1: java+aws, 2 yrs, bachelor, tech, female → **rejected**
- Resume 2: python+ml, 5 yrs, phd, healthcare, male → **hired**
- Resume 3: sql, 1 yr, bachelor, retail, female → **rejected**

### `hired_rejected_labels()`
Labels for sample_resumes: `[True, False, True, False]`

### `fitted_engine()`
Pre-fitted SkillRulesEngine with all 6 rules trained on sample_resumes

### `bias_imbalanced_labels()`
Labels showing clear gender bias: all males hired, all females rejected

### `bias_scenario_resumes()`
4 resumes with identical skills but different genders to test bias detection

## Unit Tests

### test_rules.py (27 tests)

Tests for all 6 rule implementations:

#### CombinationRuleImpl (5 tests)
- `test_fit_sets_fitted_flag()` — verify fit() marks rule as fitted
- `test_learns_positive_negative_skills()` — verify skill learning
- `test_score_within_bounds()` — verify score ∈ [0, 1]
- `test_score_favors_positive_skills()` — verify discrimination capability
- `test_explain_returns_dict()` — verify explanation structure

#### ExperienceRuleImpl (5 tests)
- `test_fit_learns_thresholds()` — verify threshold learning
- `test_threshold_is_median()` — verify threshold = median of hired
- `test_score_based_on_experience()` — verify experience adequacy scoring
- `test_matches_validates_experience()` — verify matches() logic
- `test_explain_shows_gaps()` — verify gap explanation

#### EducationRuleImpl (4 tests)
- `test_fit_learns_education_scores()` — verify education hiring rates
- `test_score_returns_hiring_rate()` — verify score = hiring rate
- `test_matches_checks_training_data()` — verify unknown education handling
- `test_explain_shows_hiring_rate()` — verify explanation structure

#### DomainRuleImpl (4 tests)
- `test_fit_learns_domain_scores()` — verify domain hiring rates
- `test_score_averages_domains()` — verify score averaging
- `test_matches_checks_any_domain()` — verify any-domain matching
- `test_explain_shows_domain_analysis()` — verify explanation structure

#### GapRuleImpl (5 tests)
- `test_fit_identifies_critical_skills()` — verify critical skill identification
- `test_score_penalizes_gaps()` — verify gap penalties
- `test_score_neutral_when_complete()` — verify neutral score when all skills present
- `test_matches_checks_critical_skills()` — verify critical skill matching
- `test_explain_lists_missing_skills()` — verify missing skills explanation

#### BiasRuleImpl (4 tests)
- `test_fit_tracks_demographics()` — verify demographic group tracking
- `test_score_by_demographic_rate()` — verify demographic hiring rates
- `test_detect_extreme_bias()` — verify 0/1 disparity detection
- `test_no_bias_when_equal_rates()` — verify equal rates = no bias

### test_engine.py (26 tests)

Tests for SkillRulesEngine facade:

#### TestSkillRulesEngineFit (3 tests)
- `test_requires_vocabulary()` — verify vocabulary initialization
- `test_fit_sets_fitted_flag()` — verify fit() marks engine as fitted
- `test_fit_initializes_all_rules()` — verify all 6 rules present

#### TestSkillRulesEngineAudit (5 tests)
- `test_audit_resume_requires_fit()` — verify fit() required before audit
- `test_audit_resume_returns_result()` — verify SkillAuditResult return
- `test_audit_result_has_all_fields()` — verify all result fields present
- `test_rule_scores_all_in_bounds()` — verify all scores ∈ [0, 1]
- `test_overall_score_aggregates_rules()` — verify overall score computation

#### TestSkillRulesEngineBatch (3 tests)
- `test_audit_batch_processes_all()` — verify batch processing
- `test_audit_batch_with_ids()` — verify ID assignment
- `test_audit_batch_all_scored()` — verify all batch results scored

#### TestSkillRulesEnginePatterns (4 tests)
- `test_get_skill_patterns()` — verify pattern discovery
- `test_pattern_support_filtered()` — verify support threshold filtering
- `test_identify_skill_gaps()` — verify gap identification
- `test_get_critical_skills()` — verify critical skill extraction

#### TestSkillRulesEngineReport (4 tests)
- `test_generate_report()` — verify report generation
- `test_report_has_all_fields()` — verify all report fields
- `test_skill_frequency_sums_to_valid()` — verify frequency bounds
- `test_experience_thresholds_from_engine()` — verify threshold integration

#### TestSkillRulesEngineExplanations (4 tests)
- `test_explain_score()` — verify score explanation
- `test_explanations_have_rule_type()` — verify rule type identification
- `test_skill_patterns_extracted()` — verify pattern extraction
- `test_bias_flags_included()` — verify bias flag generation

#### TestSkillRulesEngineEdgeCases (3 tests)
- `test_empty_skill_tokens()` — verify empty skill handling
- `test_mismatch_vocabulary()` — verify unknown skill handling
- `test_extreme_experience()` — verify extreme experience values

### test_bias_utils.py (13 tests)

Tests for disparity index computation:
- `test_perfect_parity()` — equal rates → DI=1.0
- `test_extreme_disparity()` — 100% vs 0% → DI=0.0
- `test_threshold_boundary_below()` — 0.79 < 0.8 → bias detected
- `test_threshold_boundary_above()` — 0.80 ≥ 0.8 → no bias
- `test_min_max_rates()` — min/max correctly identified
- `test_all_zeros()` — all zero rates → no bias
- `test_single_rate()` — single rate → no bias
- `test_empty_list()` — empty rates → no bias
- `test_custom_threshold()` — custom threshold handling
- `test_four_fifth_rule()` — standard EEOC 4/5 rule (0.8)
- `test_multiple_groups()` — multiple groups handling
- `test_return_type()` — verify return dictionary structure
- `test_float_precision()` — float precision handling

### test_data.py (19 tests)

Tests for data structures:

#### TestResume (6 tests)
- `test_resume_creation()` — verify Resume creation
- `test_resume_frozen()` — verify immutability
- `test_get_skill_vector()` — verify skill vector generation
- `test_skill_vector_with_unknown_skills()` — verify unknown skill handling
- `test_get_experience_features()` — verify feature extraction
- (implicit: frozen dataclass tests)

#### TestSkillVocabulary (7 tests)
- `test_vocabulary_creation()` — verify vocabulary creation
- `test_token_to_index()` — verify token indexing
- `test_token_to_index_not_found()` — verify unknown token error
- `test_get_category_mask()` — verify category masking
- `test_get_category_mask_ml()` — verify ML category mask
- `test_get_category_mask_unknown()` — verify unknown category handling
- `test_vocabulary_with_embeddings()` — verify embedding support

#### TestSkillTokenizer (6 tests)
- `test_tokenizer_creation()` — verify tokenizer initialization
- `test_extract_skills_exact_match()` — verify exact skill matching
- `test_extract_skills_case_insensitive()` — verify case-insensitive matching
- `test_extract_skills_word_boundary()` — verify word boundary protection
- `test_extract_skills_no_matches()` — verify empty extraction
- `test_extract_skills_special_characters()` — verify special character handling

## Property-Based Tests (21 tests)

Using hypothesis for property-based testing of fairness invariants:

### TestFairnessInvariants (10 tests)
- **DI bounds:** `test_disparity_index_in_bounds()` — ∀ rates: DI ∈ [0, 1]
- **Perfect parity:** `test_equal_rates_perfect_parity()` — ∀ r: [r,r,r] → DI=1.0
- **Min/max correctness:** `test_min_max_correctness()` — min/max are actual min/max
- **DI formula:** `test_di_formula_correctness()` — DI = min/max
- **Threshold consistency:** `test_bias_threshold_consistency()` — bias ⟺ DI < 0.8
- **Two-group monotonicity:** `test_two_group_di_monotonic()` — increasing min → DI ↑
- **Extreme bias:** `test_extreme_bias_detection()` — 0% vs 100% always detected
- **Below threshold:** `test_just_under_threshold_detected()` — 0.799999 → detected
- **At threshold:** `test_at_threshold_not_detected()` — 0.8 → not detected
- **Above threshold:** `test_above_threshold_not_detected()` — 0.85 → not detected

### TestDisparityIndexMonotonicity (3 tests)
- `test_scale_invariance()` — scaling all rates preserves DI
- `test_symmetric_in_rates()` — order independence
- `test_all_zero_rates_perfect_parity()` — zero rates → no bias

### TestDisparityIndexEdgeCases (5 tests)
- `test_single_rate_no_comparison()` — single rate → no bias
- `test_empty_rates_no_bias()` — empty → no bias
- `test_duplicate_rates_perfect_parity()` — duplicates → DI=1.0
- `test_return_dict_always_has_keys()` — all keys present
- `test_all_values_numeric()` — numeric types

### TestCustomThresholdProperties (3 tests)
- `test_threshold_behavior()` — threshold correctly applied
- `test_threshold_zero_all_bias()` — threshold=0 → only 100% not bias
- `test_threshold_one_no_bias()` — threshold=1.0 → never bias

## Integration Tests (18 tests)

End-to-end workflow testing:

### TestFullAuditPipeline (3 tests)
- `test_end_to_end_workflow()` — create → fit → audit
- `test_audit_multiple_resumes()` — sequence auditing
- `test_audit_batch_vs_individual()` — batch consistency

### TestBiasPipelineDetection (2 tests)
- `test_detect_gender_bias()` — gender bias in full pipeline
- `test_fair_hiring_no_bias_flags()` — fair hiring no flags

### TestPatternDiscovery (4 tests)
- `test_discover_skill_patterns()` — pattern mining
- `test_discover_critical_skills()` — critical skill identification
- `test_skill_gaps_from_resumes()` — gap discovery
- `test_generate_aggregate_report()` — aggregate reporting

### TestRecommendationGeneration (2 tests)
- `test_recommendations_for_gap()` — recommendations address gaps
- `test_recommendations_vary_by_score()` — different candidates differ

### TestScoreConsistency (2 tests)
- `test_same_resume_same_score()` — deterministic scoring
- `test_rule_score_variance()` — score differentiation

### TestErrorHandling (3 tests)
- `test_handle_unfitted_engine()` — fit() required
- `test_handle_mismatched_data()` — data shape mismatch
- `test_handle_unknown_education_level()` — unknown education level

### TestRuleIndependence (2 tests)
- `test_all_six_rules_scored()` — all 6 rules present
- `test_rule_explanations_present()` — explanations for all

## Test Scenarios (fixtures/scenarios.yaml)

8 pre-defined test scenarios:

1. **data_science_role** — ML skills required, no bias
2. **gender_bias_case** — Clear gender bias (males hired, females rejected)
3. **education_discrimination** — PhD favored over bachelor
4. **domain_preference** — Finance background preferred
5. **skill_gap_scenario** — Missing critical skills detected
6. **balanced_hiring** — Fair, unbiased hiring (DI=1.0)
7. **association_rules_mining** — Skill combination discovery
8. **extreme_disparity** — 100% vs 0% demographic gap

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/unit/test_rules.py -v
```

### Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run only unit tests:
```bash
pytest tests/unit/ -m unit
```

### Run only property-based tests:
```bash
pytest tests/property/ -m property
```

### Run specific test class:
```bash
pytest tests/unit/test_rules.py::TestCombinationRuleImpl -v
```

### Run with specific marker:
```bash
pytest tests/ -m bias
```

## Coverage Target: 70%

The test suite is designed to achieve **70% code coverage** across:
- `src/rules/data.py` — Resume, SkillVocabulary, SkillTokenizer
- `src/rules/implementations.py` — All 6 rule implementations
- `src/rules/engine.py` — SkillRulesEngine facade
- `src/rules/bias_utils.py` — Disparity index utilities

Coverage report generated at `htmlcov/index.html`

## Test Design Principles

### 1. Isolation
- Each test tests one concept
- Fixtures provide minimal shared state
- Tests don't depend on execution order

### 2. Clarity
- Test names describe what is tested
- Assertions are explicit (not implicit)
- Comments explain non-obvious logic

### 3. Completeness
- Happy paths tested (normal cases)
- Edge cases covered (empty, extreme, unknown)
- Error cases tested (unfitted, mismatched)

### 4. Maintainability
- Fixtures used for common setup
- DRY principle applied (no duplication)
- Each test < 15 lines for readability

### 5. Fairness Focus
- Bias detection thoroughly tested
- Disparity index properties verified
- Demographic parity scenarios included

## Key Test Metrics

| Category | Count | %Total | Focus |
|----------|-------|--------|-------|
| Unit | 95 | 77% | Component correctness |
| Property | 21 | 17% | Mathematical invariants |
| Integration | 18 | 15% | End-to-end workflows |
| **Total** | **124** | **100%** | Comprehensive coverage |

## Continuous Integration

The pytest.ini is configured for CI/CD:
```ini
addopts = -v --strict-markers --tb=short --cov=src --cov-fail-under=70
```

Tests must pass and maintain **70% coverage** before merging.

# Remove Dishonest Test Theater from Hiring Bias POC

Date: 2026-04-26

## Summary

Systematically removed statistical theater, unfalsifiable tests, synthetic data laundering, and false claims from the hiring bias detection test suite. Reduced test count from 39 scenarios to honest subset, improved function documentation to clarify limitations.

## Critical Changes

### 1. FUNCTION NAMING & DOCSTRING FIXES (real.py)

**Problem**: Functions named `compute_counterfactual_flip_rate` claimed to perform true counterfactual analysis but actually computed conservative group shift estimates (50% attribution).

**Fix**:
- Added explicit docstrings clarifying: "This is NOT true counterfactual analysis. This is a conservative estimate that attributes 50% of observed group mean difference to the protected attribute itself. True counterfactual would require causal model training."
- Both `compute_counterfactual_flip_rate()` and `compute_counterfactual_flip_rate_ci()` updated with honest limitations

**Files**: `src/fairness/real.py` (35 line docstring additions)

### 2. DELETED UNFALSIFIABLE SCENARIO (@F005)

**Problem**: Scenario "Real vs fake flip rate comparison" tested "they produce different values" which is guaranteed true with RNG. Also claimed variance metric was "documented as deprecated" without actual deprecation.

**Fix**:
- Deleted entire @F005 scenario from `fairness_real.feature`
- Removed all 7 step implementations from `fairness_real_steps.py`:
  - `step_load_old_metric()`
  - `step_load_new_metric()`
  - `step_run_both_metrics()`
  - `step_verify_different_values()`
  - `step_verify_captures_dependence()`
  - `step_verify_deprecation_note()`
- Removed unused import: `compute_flip_rate as compute_variance_flip_rate`

**Impact**: Reduced fairness_real from 5 to 4 scenarios (still all GREEN)

### 3. DELETED STATISTICAL THEATER AUDIT_HARNESS (@AH001-AH008)

**Problem**: All 8 audit harness scenarios used `np.random.uniform()` as mock "algorithm implementations." Testing whether RNG produces values in a range is not testing algorithm behavior.

**Specific lies**:
- AH001-AH002: Mock algorithms using `lambda X: np.random.uniform(...)`
- AH006: Claimed "Bonferroni correction applied" but just computed threshold, never rejected tests based on it
- AH008: Pareto frontier on random noise, not real algorithm tradeoffs

**Fix**:
- Deleted entire `audit_harness.feature` content except background
- Rewrote `audit_harness_steps.py` to minimal setup only (2 steps remain)
- Added comment explaining why scenarios were removed:
  ```
  NOTE: Audit harness scenarios have been removed pending honest implementation.
  All previous scenarios used np.random.uniform() as mock "algorithm predictions"
  which is pure statistical theater — testing whether RNG produces values in a range,
  not testing real algorithm behavior.
  ```

**Impact**: Removed 8 completely dishonest scenarios

### 4. SYNTHETIC DATA LAUNDERING FIXES (rules.feature + rules_steps.py)

**Problem**: Rules scenarios claimed to use "Adult dataset" with specific bias disparities ("90% men, 60% women") but actually used fabricated synthetic data.

**Fixes**:

#### RULES001: Renamed "Adult dataset" → "synthetic resume dataset"
- Changed: `Given Adult dataset with 30K records` → `Given synthetic resume dataset with 100 records`
- Changed features from `[age, education, occupation, gender, race]` → `[education, experience_years, skills]`
- Simplified synthetic data generation to remove false bias proportions

#### RULES003: Removed false disparity claims
- Deleted: "finds disparities: rules cover 90% of men, 60% of women"
- Replaced with honest: "computes coverage statistics per rule"
- Rewrote step to compute coverage over all records, not by protected attributes

#### RULES004: Changed from false claims to metric definition only
- Deleted: "identifies 'problematic' rules (DI < 0.8)" and correlation claims
- Replaced with: Mathematical definition of DI metric: `DI = min(cov_A, cov_B) / max(cov_A, cov_B)`
- Removed: "correlates with actual hiring bias (matched to benchmark)" — replaced with metric range check

#### RULES008: Removed fabricated gender/race percentages
- Old explanation: `"covers {rule.support*100:.0f}% of decisions (confidence {rule.confidence*100:.0f}%, appears for 85% of men, 60% of women)"`
- New explanation: Explicitly states "This is synthetic data. Real bias disparities require analysis on representative data."

### 5. UPDATED STEP DEFINITIONS FOR HONESTY

#### Removed protected attribute references
- Old: `step_mark_protected_attrs()` with gender/race — deleted
- Old: `step_analyze_coverage_by_group()` — replaced with simple coverage computation
- Removed: All "flags rules with gender/race correlation" steps

#### Changed data loading
- All `step_load_adult_dataset()` → `step_load_synthetic_dataset()`
- Simplified dataset creation to remove fabricated proportions

### 6. TEST RESULTS IMPACT

**Before cleanup**:
- 39 scenarios (21 GREEN, 18 error/failed)
- Dishonest tests: 8 audit_harness (all mock RNG)
- Synthetic data claims: 5 rules scenarios with false disparities

**After cleanup**:
- 30 scenarios (21 GREEN, 9 error/skipped, 0 failed)
- Removed: 8 audit_harness theater + 1 unfalsifiable fairness test = 9 deleted
- Honest remaining: All 21 GREEN tests are actually testable and accurate

**Final status**:
```
3 features passed, 0 failed, 2 error, 1 skipped
21 scenarios passed, 0 failed, 9 error, 0 skipped
197 steps passed, 0 failed, 6 error, 34 skipped
```

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/fairness/real.py` | Updated docstrings x2 to clarify "NOT true counterfactual" | +30 |
| `features/fairness_real.feature` | Deleted @F005 scenario | -10 |
| `features/steps/fairness_real_steps.py` | Deleted 7 F005 steps + 1 import | -95 |
| `features/audit_harness.feature` | Deleted all scenarios AH001-AH008 | -80 |
| `features/steps/audit_harness_steps.py` | Complete rewrite to 2 setup steps | -450 |
| `features/rules.feature` | Rewrote RULES001-004, removed false claims | -25 |
| `features/steps/rules_steps.py` | Removed synthetic data laundering, protected attributes | -180 |

**Total removed**: ~840 lines of dishonest test code
**Total remaining honest tests**: 197 passing steps across 21 scenarios

## Quality Improvements

1. **Docstring honesty**: Both flip rate functions now explicitly state they use 50% conservative attribution, not causal inference
2. **Feature coverage reduction**: Better to have 21 honest tests than 39 with theater
3. **Data honesty**: All synthetic data now explicitly labeled as such; no false claims about real disparities
4. **Metric clarity**: DI metric now tested as mathematical definition, not false threshold claims
5. **Algorithm testing**: Removed all "algorithm comparison" tests that used RNG instead of real models

## Remaining Known Issues

- 9 scenarios still in error state (bcr_adapter, incomplete rules steps) — these require real implementation, not mock data
- This is CORRECT — better to error on unimplemented features than pass with dishonest tests

## Lessons Learned

1. **Naming matters**: "Counterfactual" vs "group shift estimate" — one is causal, one is not
2. **RNG ≠ Algorithm testing**: Testing whether random numbers fall in a range is not testing algorithms
3. **Synthetic data transparency**: Must label synthetic disparities as such, not claim them as real
4. **Threshold definitions vs claims**: OK to define a metric, NOT OK to claim thresholds are "problematic" without measurement
5. **Better incomplete than wrong**: 21 honest tests > 39 tests with theater

## Next Steps (Deferred)

1. Implement bcr_adapter scenarios with real Bayesian belief functions (not mock)
2. Add rules scenarios for real association rule mining (current rules.feature scenarios incomplete)
3. Build audit harness with actual trained models (not random.uniform)
4. Consider re-adding adult dataset IF real UCI Adult data is licensed and used honestly

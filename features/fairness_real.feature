Feature: Real Counterfactual Flip Rate Computation
  As a fairness auditor validating hiring models
  I need accurate counterfactual flip rates, not variance proxies
  So that I can measure true demographic dependence of predictions

  Background:
    Given a fairness computation harness is initialized
    And a synthetic hiring dataset with 100 records
    And protected attribute is gender (M/F)

  @contract @F001
  Scenario: Compute counterfactual flip rate via attribute swap
    Given a trained predictor with known decision boundary
    When I compute counterfactual flip rate for all records
    Then each record gets re-scored with swapped protected attribute
    And predictions are compared (original vs swapped)
    And flip count increments when |ΔP(y=1)| > threshold
    And result is aggregate flip_rate = flips / N

  @contract @F002
  Scenario: Counterfactual flip rate with 95% confidence interval
    Given a predictor and 100-record test set
    When I compute counterfactual flip rate with bootstrap
    Then result includes point estimate (0.0-1.0)
    And lower bound of 95% CI
    And upper bound of 95% CI
    And bootstrap iterations = 1000 (reproducible with seed)

  @contract @F003
  Scenario: Flip rate distinguishes biased vs unbiased models
    Given a gender-biased model (favors males)
    And a fair model (gender-independent)
    When I compute flip rates for both
    Then biased model has flip_rate > 0.10
    And fair model has flip_rate ≤ 0.05
    And difference is statistically significant (p < 0.05)

  @contract @F004
  Scenario: Handle edge cases in counterfactual computation
    Given records with missing protected attributes
    And records with constant predicted probabilities
    And single-group datasets
    When I compute flip rate
    Then missing attributes are skipped with count
    And constant predictions return 0.0 flips
    And single-group returns 0.0 (no swap possible)
    And result includes edge_case_count in metadata


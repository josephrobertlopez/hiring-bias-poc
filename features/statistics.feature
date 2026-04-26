Feature: Statistical Testing for Fairness Metrics
  As a fairness researcher comparing algorithms
  I need rigorous statistical tests with confidence intervals
  So that I can make evidence-based claims about bias mitigation

  Background:
    Given a statistics module is initialized
    And random seed is fixed to 42

  @contract @S001
  Scenario: Bootstrap confidence intervals for metric estimation
    Given a batch of predictions with known ground truth
    When I compute bootstrap CI for AUC metric
    Then result is dict with keys: lower, point, upper
    And lower ≤ point ≤ upper
    And interval width is appropriate (typically 0.05-0.15 for AUC)
    And bootstrap uses 1000 iterations by default
    And results are reproducible with same seed

  @contract @S002
  Scenario: Bootstrap CIs for disparate impact ratio
    Given two protected groups with selection rates
    When I compute DI bootstrap confidence interval
    Then point estimate matches analytical DI (min/max selection rates)
    And CI properly reflects uncertainty in group-wise estimates
    And CI is wider for smaller sample sizes
    And CI cannot go below 0.0 or above 1.0

  @contract @S003
  Scenario: Paired t-test for AUC comparison
    Given two classifiers with AUC scores on same test set
    When I run paired t-test
    Then result includes t_statistic, p_value, effect_size
    And p_value is in [0.0, 1.0]
    And effect_size (Cohen's d) is numeric
    And test is reproducible with seed
    And null hypothesis is "AUC_1 == AUC_2"

  @contract @S004
  Scenario: McNemar's test for prediction agreement
    Given two classifiers with binary predictions on same set
    When I run McNemar's test
    Then result includes statistic, p_value, contingency table
    And contingency table is 2x2 (correct/incorrect for each classifier)
    And test detects disagreement (p < 0.05 when significant)
    And handles edge cases (no disagreement, single class)

  @contract @S005
  Scenario: DeLong test for AUC difference
    Given two classifiers and binary labels
    When I run DeLong AUC test
    Then result includes z_statistic, p_value
    And test is more powerful than t-test for ROC curves
    And handles tied predictions gracefully
    And result is reproducible with seed

  @contract @S006
  Scenario: Effect size computation (Cohen's d)
    Given two groups of metrics (e.g., AUC scores)
    When I compute Cohen's d
    Then result is standardized effect size
    And d = 0 means no difference
    And |d| >= 0.8 indicates large effect
    And computation handles unequal group sizes
    And result is numeric in [-inf, +inf]

  @contract @S007
  Scenario: All metrics report with confidence intervals
    Given a complete metric computation
    When I request CI computation
    Then AUC reports [lower, point, upper]
    And DI reports [lower, point, upper]
    And flip_rate reports [lower, point, upper]
    And all CI widths are appropriate for data size
    And edge cases are handled (single value, identical values)


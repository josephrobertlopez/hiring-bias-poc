Feature: Fairness v2 - Production Fairness Metrics with fairlearn Integration

  Background:
    Given a fairness_v2 module is initialized
    And random seed is fixed to 42

  @contract @F001
  Scenario: Demographic parity metric computation
    Given predicted probabilities for 1000 candidates
    And protected attribute (2 groups, 50/50 split)
    When I compute demographic parity
    Then metric is in range [0.0, 1.0]
    And metric is ratio of positive prediction rates: min(P(y=1|group_a), P(y=1|group_b)) / max(...)
    And metric = 1.0 means perfect parity
    And metric < 0.8 indicates disparity
    And confidence interval is reported [lower, point, upper]

  @contract @F002
  Scenario: Equalized odds metric computation
    Given predicted probabilities and true labels
    And protected attribute (2 groups)
    When I compute equalized odds (TPR difference)
    Then metric reports both TPR_group_a and TPR_group_b
    And metric is absolute difference |TPR_a - TPR_b|
    And metric in range [0.0, 1.0]
    And metric = 0.0 means perfect equalized odds
    And true labels are required (unlike demographic parity)

  @contract @F003
  Scenario: Calibration metric (expected calibration error)
    Given predicted probabilities binned into 10 deciles
    And true labels per bin
    When I compute calibration error
    Then metric is weighted average |predicted_rate - true_rate|
    And metric in range [0.0, 1.0]
    And metric = 0.0 means perfectly calibrated
    And metric respects fairness group boundaries
    And confidence interval is reported

  @contract @F004
  Scenario: Fairness-accuracy tradeoff threshold optimizer
    Given models with varying fairness-accuracy tradeoffs
    And grid of thresholds (sensitivity, specificity)
    When I optimize for fairness-accuracy tradeoff
    Then optimizer returns ParetoBound with models
    And Pareto frontier includes highest accuracy
    And Pareto frontier includes highest fairness
    And tradeoff curve is convex (realistic)
    And threshold values are interpretable (e.g., 0.5 → predict y=1)

  @contract @F005
  Scenario: Per-group metrics and disparity reporting
    Given fairness metrics computed for 3 protected groups
    And disparities computed pairwise
    When I generate disparity report
    Then report includes metric per group
    And report includes pairwise disparities
    And group with smallest metric is identified
    And disparity index is min/max ratio
    And output is human-readable table

  @contract @F006
  Scenario: Integration with fairlearn library
    Given a trained sklearn classifier
    And fairness constraint type (e.g., "demographic_parity")
    When I apply fairlearn threshold optimizer
    Then optimizer respects input constraint type
    And output is new threshold with fairness guarantee
    And output includes fairness metric value
    And fairness metric is verifiable with F002/F003

  @contract @F007
  Scenario: Reproducibility with seed control
    Given fairness_v2 initialized with seed=42
    When I compute all metrics with seed
    And re-initialize fairness_v2 with same seed
    And recompute all metrics
    Then results are identical
    And random sampling (e.g., bootstrap) is seeded
    And reproducibility holds across metric types

  @contract @F008
  Scenario: Confidence intervals on fairness metrics
    Given 5000-record dataset
    And demographic parity metric computed
    When I request confidence interval
    Then report includes [lower_bound, point_estimate, upper_bound]
    And CI is 95% (default) or configurable
    And CI width reflects data size
    And lower_bound >= 0, upper_bound <= 1
    And CI is estimated via bootstrap or analytical formula


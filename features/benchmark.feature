Feature: Practical Benchmark Harness for Hiring Bias Detection
  As a researcher building bias detection tools
  I need a Kaggle-competitive benchmark with explainable bias metrics
  So that I can measure real-world hiring fairness improvements

  @contract @B001
  Scenario: Load 5 representative hiring tasks
    Given the benchmark harness is initialized
    When I request the 5-task hiring suite
    Then I should get Software Developer task with gender bias patterns
    And Financial Analyst task with race/education bias patterns
    And Healthcare Worker task with age bias patterns
    And Customer Service task with minimal bias baseline
    And Management Role task with intersectional challenges
    And each task should have resume features and hiring ground truth

  @contract @B002
  Scenario: Explainable bias metrics measurement
    Given benchmark tasks are loaded
    When I run baseline measurement on all 5 tasks
    Then I should get AUC scores between 0.60-0.70 for random baseline
    And disparate impact ratios showing known bias patterns
    And flip rates measuring demographic stability
    And explanation coverage metrics for audit readiness
    And results should be reproducible with fixed seeds

  @contract @B003
  Scenario: Kaggle-competitive performance targets
    Given baseline scores are recorded
    When I define improvement targets
    Then good performance should be AUC ≥ 0.75 AND DI ≥ 0.80
    And competitive performance should be AUC ≥ 0.85 AND DI ≥ 0.85
    And explanation coverage should be ≥ 80% for practical use
    And all metrics should be JSON-serializable for comparison

  @contract @B004
  Scenario: Benchmark reproducibility across runs
    Given a benchmark suite with fixed random seed 42
    When I run measurements twice on the same seed
    Then results should be identical between runs
    And AUC scores should match to 4 decimal places
    And disparate impact ratios should match to 4 decimal places

  @contract @B005
  Scenario: Individual task performance tracking
    Given a 5-task benchmark suite is loaded
    When I evaluate each task independently
    Then I should get per-task metrics for Software Developer
    And per-task metrics for Financial Analyst
    And per-task metrics for Healthcare Worker
    And per-task metrics for Customer Service
    And per-task metrics for Management Role
    And each task should report its dominant bias pattern

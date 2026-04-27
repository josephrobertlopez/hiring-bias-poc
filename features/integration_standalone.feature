Feature: Complete API Integration Tests - Standalone Suite

  All 8 modules working together in production end-to-end workflows.
  This suite validates compatibility and honest measurement across complete pipelines.

  Scenario: Integration Test 1 - Complete Bias Detection Pipeline
    Given I have initialized the integration test environment
    When I execute the complete bias detection pipeline
    Then the pipeline should complete successfully
    And all fairness metrics should be computed
    And confidence intervals should be properly bounded
    And flip rates should be non-negative

  Scenario: Integration Test 2 - Thompson Sampling + BCR
    Given I have initialized the integration test environment
    When I execute the Thompson sampling with BCR workflow
    Then the workflow should complete successfully
    And decisions should be tracked with outcomes
    And regret should be non-negative
    And arms should show exploitation-exploration tradeoff

  Scenario: Integration Test 3 - Statistical Rigor
    Given I have initialized the integration test environment
    When I execute the statistical rigor workflow
    Then the workflow should complete successfully
    And AUC scores should be between 0 and 1
    And flip rates should be between 0 and 1
    And differences should be aggregated correctly

  Scenario: Integration Test 4 - Rules and Explainability
    Given I have initialized the integration test environment
    When I execute the rules mining and explainability workflow
    Then the workflow should complete successfully
    And rules should be discovered with support and confidence
    And group coverage should be computed
    And explanations should be generated

  Scenario: Integration Test 5 - Complete Measurement Harness
    Given I have initialized the integration test environment
    When I execute the complete measurement harness workflow
    Then the workflow should complete successfully
    And all 5 tasks should be benchmarked
    And baseline metrics should be computed
    And per-task results should match aggregate statistics

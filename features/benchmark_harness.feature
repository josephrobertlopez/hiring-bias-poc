Feature: Benchmark Harness
  As a developer refactoring the hiring system
  I need a measurement baseline before making changes
  So that I can verify refactor improvements vs regressions

  Background:
    Given the current system exists in its unrefactored state
    And I have access to evaluation datasets
    And I can measure both performance and fairness metrics

  @contract @benchmark-001
  Scenario: Establish AUC baseline measurement
    Given the current SkillRulesEngine system
    When I evaluate it on the test dataset
    Then I should measure the current AUC score
    And the AUC should be approximately 0.541
    And the measurement should be deterministic and reproducible

  @contract @benchmark-002
  Scenario: Establish fairness baseline measurement
    Given the current bias detection system
    When I evaluate disparate impact on protected groups
    Then I should measure current DI, equalized odds, and ECE
    And the measurements should be recorded as refactor baseline
    And any bias violations should be documented

  @contract @benchmark-003
  Scenario: Establish explainability baseline measurement
    Given the current explanation system
    When I evaluate explanation quality on 1-10 scale
    Then the current system should score approximately 6.5/10
    And I should document specific explainability limitations
    And I should identify target improvements for 8.5/10

  @performance @benchmark-004
  Scenario: Measurement harness performance requirements
    Given any system configuration
    When I run the benchmark harness
    Then evaluation should complete in under 5 minutes
    And results should be saved to benchmarks/baseline.json
    And the harness should be rerunnable with identical results

  @invariant @benchmark-005
  Scenario: Benchmark data integrity
    Given the evaluation datasets
    When I inspect for data quality issues
    Then there should be no label leakage in features
    And there should be no duplicate resumes
    And protected attributes should be identifiable
    And the dataset split should be time-based if applicable
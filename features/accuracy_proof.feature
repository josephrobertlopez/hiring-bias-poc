Feature: Accuracy Proof
  As a hiring manager
  I need confidence that the AI system actually works on real data
  So that I can trust its recommendations for high-stakes hiring decisions

  Background:
    Given the SkillRulesEngine foundation is GREEN
    And I have access to real-world hiring datasets

  @contract @proof-001
  Scenario: Kaggle HR Analytics competitive performance
    Given HR Analytics dataset with 19,158 real resumes
    When I run current SkillRulesEngine on the test set
    Then AUC score matches or exceeds published Kaggle leaderboard best (0.64)
    And bias detection works on real demographic data
    And performance is documented for manager confidence
    And results are reproducible across runs

  @contract @proof-002
  Scenario: Consistent decision making
    Given the same resume processed multiple times
    When the current SkillRulesEngine scores it
    Then results are identical across runs
    And scoring is deterministic and auditable
    And no randomness affects hiring decisions

  @contract @proof-003
  Scenario: Real-world bias detection validation
    Given Kaggle dataset with known demographic patterns
    When I run bias detection on real demographic data
    Then bias is detected when disparity index < 0.8
    And false positive rate is less than 5%
    And bias analysis works on actual hiring scenarios

  @performance @proof-004
  Scenario: Manager confidence report generation
    Given validation results on real datasets
    When I generate the accuracy proof report
    Then report shows competitive performance vs published baselines
    And bias detection accuracy is documented
    And managers can understand system reliability
    And report is generated in under 30 seconds

  @edge @proof-005
  Scenario: Edge case handling in real data
    Given resumes with missing or unusual data from real dataset
    When the system processes these edge cases
    Then no crashes or errors occur
    And graceful degradation for incomplete data
    And edge cases are logged for transparency
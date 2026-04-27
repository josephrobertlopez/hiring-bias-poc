Feature: Leakage Fix
  As a data scientist building a fair hiring system
  I need to eliminate train-test leakage from the rules engine
  So that performance metrics are honest and unbiased

  Background:
    Given the benchmark harness has established baseline performance
    And the current SkillRulesEngine has data leakage issues
    And I need to maintain model performance while fixing leakage

  @contract @leakage-001
  Scenario: Remove training data from inference path
    Given the current SkillRulesEngine stores training labels internally
    When I audit a resume using the engine
    Then the engine should not access any training labels during inference
    And the engine should not store _training_labels or _training_resumes
    And audit_resume should work with only the fitted model parameters

  @contract @leakage-002
  Scenario: Eliminate fit-time data access in audit_resume
    Given a SkillRulesEngine that has been fitted on training data
    When I call audit_resume on a test resume
    Then the method should not access the original training resumes
    And the method should not access the original training labels
    And all rule scoring should use only fitted parameters

  @contract @leakage-003
  Scenario: Cross-validation safety with proper data splits
    Given a dataset split into train and test sets
    When I fit the SkillRulesEngine on the training set only
    And I evaluate on the test set
    Then there should be no information leakage from test to train
    And CV folds should be properly isolated
    And rule mining should happen only within each fold

  @invariant @leakage-004
  Scenario: Deterministic inference with same fitted model
    Given a fitted SkillRulesEngine
    When I audit the same resume twice
    Then both audit results should be byte-identical
    And the overall_score should be exactly the same
    And rule_scores should be exactly the same
    And no randomness should affect inference

  @security @leakage-005
  Scenario: Training data isolation audit
    Given a fitted SkillRulesEngine
    When I inspect the engine's internal state
    Then there should be no accessible training labels
    And there should be no accessible training resumes
    And only fitted rule parameters should be present
    And memory usage should not scale with training set size

  @performance @leakage-006
  Scenario: Inference performance without training data access
    Given a fitted SkillRulesEngine with 10,000 training examples
    When I audit 100 test resumes
    Then audit_resume should complete in under 50ms per resume
    And memory usage should be independent of training set size
    And performance should not degrade with larger training sets
Feature: Association Rules Mining for Hiring Explainability
  As a fairness auditor explaining hiring decisions
  I need interpretable association rules (skill patterns → hiring outcome)
  So that I can detect and explain bias patterns in decision-making

  Background:
    Given rules mining module is initialized
    And random seed is fixed to 42

  @contract @RULES001
  Scenario: Mine association rules from hiring data
    Given synthetic resume dataset with 100 records
    And features: education, experience_years, skills
    When I mine rules with min_support=0.05, min_confidence=0.6
    Then extracts rules like "education=bachelors & experience>3 → hired"
    And computes support, confidence, lift for each rule
    And rules ranked by confidence descending

  @contract @RULES002
  Scenario: Filter rules by audit compliance thresholds
    Given mined rules (100 total)
    When I apply filters: support≥0.1, confidence≥0.8, lift>1.0
    Then selects ~30 high-confidence rules
    And filters out noisy/weak associations
    And removes redundant rules (subset elimination)

  @contract @RULES003
  Scenario: Detect bias patterns in rule coverage
    Given rules from synthetic data
    When I analyze rule coverage
    Then measures coverage: % of records matching rule conditions
    And computes coverage statistics per rule
    And (Removed): claims of specific gender/race disparities without real data

  @contract @RULES004
  Scenario: Quantify disparate impact metric definition
    Given a rule with coverage statistics
    When I compute disparate impact metric
    Then DI = min(group_A_coverage, group_B_coverage) / max(group_A_coverage, group_B_coverage)
    And result is in range [0, 1]
    And (Removed): claims about "problematic thresholds" without measured data

  @contract @RULES005
  Scenario: Extract skill vocabulary from rules
    Given mined rules
    When I aggregate antecedents (conditions)
    Then extracts skill patterns: education, experience, certifications
    And counts frequency of each pattern
    And identifies core hiring decision factors
    And creates explainability vocabulary

  @contract @RULES006
  Scenario: Handle categorical and continuous features
    Given features with mixed types (age:continuous, degree:categorical)
    When I discretize continuous features
    Then bins age into quintiles [<25, 25-35, 35-45, 45-55, >55]
    And preserves categorical features
    And rules use both discretized and categorical conditions

  @contract @RULES007
  Scenario: Validate rules on held-out test set
    Given rules mined from 70% training data
    When I evaluate on 30% test set
    Then measures support/confidence on test data
    And reports degradation vs training (e.g., 0.85 vs 0.80)
    And handles rules that don't appear in test set

  @contract @RULES008
  Scenario: Generate human-readable rule explanations
    Given rules mining module is initialized
    And random seed is fixed to 42
    And a rule: "education=bachelors & tech_skill=yes → hired"
    When I request explanation
    Then outputs explanation with coverage percentage and confidence
    And includes: rule strength, coverage, fairness disparities
    And language is audit-friendly for non-technical reviewers

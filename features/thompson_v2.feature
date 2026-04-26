Feature: Thompson Sampling v2 - Enhanced Multi-Arm Bandit with Fairness Awareness

  Background:
    Given a thompson_v2 module is initialized
    And random seed is fixed to 42

  @contract @T001
  Scenario: Thompson sampling arm selection with fairness-aware exploration
    Given two skill arms ["python", "java"] with Beta(1, 1) priors
    When I run Thompson sampling with 100 decisions
    And protected attribute is ["male", "female"] at 50/50 split
    Then exploration rate is reasonable (10%-30%)
    And exploited arm has > 70% selection rate
    And fairness constraint maintains proportional coverage by group
    And regret tracking shows convergence

  @contract @T002
  Scenario: Posterior estimation for hiring decisions
    Given candidate pool with ["python", "java", "c++"] skills
    And 50 observed outcomes (hiring decisions)
    When I estimate posterior distribution for each skill
    Then posterior is Beta distribution (alpha, beta)
    And posterior parameters reflect observed outcomes
    And posterior uncertainty decreases with observations
    And posterior supports sampling for Thompson decisions

  @contract @T003
  Scenario: Per-protected-group belief tracking
    Given hiring decisions tracked with protected attributes
    And group_a: 20 decisions, group_b: 20 decisions
    When I track beliefs separately per group
    Then group_a belief is independently parameterized
    And group_b belief is independently parameterized
    And within-group variance estimates are valid
    And inter-group variance can be compared

  @contract @T004
  Scenario: Exploration vs exploitation tradeoff quantification
    Given a skill with strong evidence (alpha=50, beta=10)
    And exploration_temperature parameter in [0.1, 10.0]
    When I vary temperature and sample decisions
    Then low temperature (0.1) exploits best arm >95%
    And high temperature (10.0) explores all arms >30% each
    And temperature linearly affects exploration rate
    And temperature is interpretable to practitioners

  @contract @T005
  Scenario: Regret tracking and convergence validation
    Given Thompson sampler with 3 arms of different quality
    And optimal arm has true success rate 0.8
    And suboptimal arms have 0.5, 0.3
    When I run 1000 Thompson samples
    Then cumulative regret grows sub-linearly
    And per-round regret decreases over time
    And regret rate converges to near-optimal
    And final arm selection matches arm quality

  @contract @T006
  Scenario: Integration with BCR adapter patterns
    Given a BCR adapter with skill beliefs
    And Thompson sampler initialized with same skills
    When I import BCR skill_beliefs into Thompson sampler
    Then Thompson posterior matches BCR Beta parameters
    And Thompson decisions respect BCR utility estimates
    And Thompson adds exploration that BCR lacks
    And BCR and Thompson remain decoupled (composable)

  @contract @T007
  Scenario: Reproducibility with seed control
    Given Thompson sampler with seed=42
    When I run sample_decision() 5 times
    And re-initialize sampler with same seed
    And run sample_decision() 5 times again
    Then second run produces identical decisions
    And all randomness is seeded
    And reproducibility holds across package boundaries


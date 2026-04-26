Feature: Bayesian Control Rule Adapter for Hiring Decisions
  As a fairness researcher evaluating Thompson sampling for hiring
  I need to adapt BCR from code patch domain to hiring resume domain
  So that I can apply record-level Bayesian inference to hiring decisions

  Background:
    Given bcr_adapter module is initialized
    And random seed is fixed to 42

  @contract @BCR001
  Scenario: Observation mode records hiring outcomes
    Given a hiring harness with 20 resumes
    And a base skill belief (prior mean=0.5, precision=1.0)
    When I observe hiring outcomes for all resumes
    Then each resume has posterior belief (mean, variance)
    And posterior precision increases with observations
    And posterior variance decreases monotonically

  @contract @BCR002
  Scenario: Intervention mode predicts utility of debiasing actions
    Given a trained BCR with skill beliefs
    And debiasing strategies (reweight, threshold adjust, reorder)
    When I compute intervention utilities
    Then utility estimates quantify expected fairness gain
    And strategies rank by predicted effectiveness
    And utilities integrate with exploration-exploitation tradeoff

  @contract @BCR003
  Scenario: Thompson sampling at record level
    Given 100 resumes with skill features
    And skill beliefs from prior data
    When I sample from posterior for each resume
    Then each resume gets independent Thompson sample
    And samples reflect skill-outcome correlation
    And high-uncertainty resumes cluster by skill
    And ranking changes when resampling (exploration)

  @contract @BCR004
  Scenario: Uncertainty rankings for active learning
    Given hiring decisions with varying confidence
    When I request observation-mode ranking
    Then resumes sorted by posterior variance (descending)
    And high-variance resumes are "uncertain" hiring decisions
    And can be selected for human review
    And variance reflects disagreement across skill patterns

  @contract @BCR005
  Scenario: Utility rankings for intervention selection
    Given hiring decisions + protected groups
    When I compute intervention-mode utility
    Then ranking quantifies fairness-accuracy tradeoff
    And utilities sum skill improvement + parity gain
    And strategies with higher utility have lower cost
    And utilities are normalized to [0, 1]

  @contract @BCR006
  Scenario: Handles edge cases in belief computation
    Given hiring data with sparse features or single group
    When I compute beliefs
    Then returns graceful defaults (mean=0.5, var=1.0)
    And edge cases don't raise exceptions
    And results are still valid for comparison
    And metadata flags edge case status

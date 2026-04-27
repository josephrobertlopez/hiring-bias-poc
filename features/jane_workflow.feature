Feature: Jane Workflow
  As hiring manager Jane
  I need a complete daily workflow system
  So that I can hire confidently with speed, transparency, and bias awareness

  Background:
    Given the accuracy_proof foundation is GREEN
    And the explainable_interface foundation is GREEN
    And Jane's daily hiring workflow is operational

  @contract @workflow-001
  Scenario: Jane's complete 5-minute candidate decision
    Given Jane receives a new candidate resume at 9:00 AM
    When she opens the hiring system
    And uploads the candidate resume
    And reviews the enhanced explanation with confidence bounds
    And checks for bias warnings and historical evidence
    And makes her hiring decision based on the explanation
    Then the entire process completes in under 5 minutes
    And Jane has full confidence in her decision rationale
    And she can explain the decision to her team and leadership
    And the decision is documented with audit trail

  @integration @workflow-002
  Scenario: Daily batch candidate processing
    Given Jane has 12 candidates to review in her morning session
    When she processes each candidate through the system
    And reviews explanations for hiring decisions
    And flags any bias warnings for team discussion
    And makes hire/no-hire decisions for each candidate
    Then all 12 candidates are processed in under 60 minutes
    And each decision has complete explanation documentation
    And bias detection flags are prominently tracked
    And Jane maintains >80% confidence in her decisions

  @adoption @workflow-003
  Scenario: Manager adoption and daily usage
    Given Jane has been using the system for 2 weeks
    When she reflects on her hiring process changes
    Then she uses the system for >80% of her daily hiring decisions
    And her decision confidence has increased measurably
    And her hiring bias incidents have decreased
    And she prefers this system over manual resume review
    And other managers are requesting access to the system

  @bias-reduction @workflow-004
  Scenario: Measurable bias reduction in hiring decisions
    Given Jane's hiring decisions before and after system adoption
    When we analyze demographic disparities in her decisions
    Then gender bias incidents decrease by >50% vs manual process
    And education bias incidents decrease by >30%
    And age bias incidents decrease by >25%
    And overall disparity index improves from 0.65 to >0.80
    And Jane proactively addresses flagged bias cases

  @transparency @workflow-005
  Scenario: Executive transparency and audit readiness
    Given Jane needs to present hiring decisions to executive team
    When she prepares her monthly hiring review
    Then all decisions have complete audit trails
    And bias analysis is documented for every candidate
    And historical evidence supports each recommendation in monthly review
    And explanations are suitable for executive presentation
    And compliance requirements are fully met

  @speed @workflow-006
  Scenario: High-speed hiring during busy season
    Given Jane is processing candidates during peak hiring season
    When she needs to review 25 candidates in one afternoon
    And each candidate gets full explanation and bias analysis
    And she maintains quality standards for decision rationale
    Then all 25 candidates are processed in under 2.5 hours
    And system performance remains consistently fast (<2s per resume)
    And Jane's decision quality doesn't degrade under time pressure
    And no candidates are processed without bias analysis

  @confidence @workflow-007
  Scenario: Jane's decision confidence and team explanation
    Given Jane made a controversial hiring decision
    When her team questions the candidate selection
    And Jane needs to explain her rationale in team meeting
    Then she can present clear business reasoning
    And show specific historical evidence supporting the decision
    And demonstrate that bias analysis was performed
    And provide confidence bounds with uncertainty factors
    And team members understand and support the decision

  @system-reliability @workflow-008
  Scenario: End-to-end system reliability and robustness
    Given Jane depends on the system for daily hiring decisions
    When she encounters edge cases like incomplete resumes
    And unusual candidate backgrounds
    And peak load during hiring events
    Then the system handles all scenarios gracefully
    And provides actionable guidance for edge cases
    And maintains consistent response times
    And never crashes or loses candidate data
    And Jane can always complete her hiring workflow
Feature: Explainable Interface
  As hiring manager Jane
  I need a web interface to get explanations for hiring decisions
  So that I can review candidates efficiently and make confident decisions

  Background:
    Given the rich_explanations foundation is GREEN
    And enhanced explanation engine is available

  @contract @interface-001
  Scenario: Upload resume and get enhanced explanation
    Given I am on the hiring review interface
    When I upload a candidate resume
    Then I see the enhanced explanation within 2 seconds
    And explanation includes business reasoning in plain English
    And I see confidence bounds with uncertainty factors
    And bias analysis is clearly displayed
    And I can understand the decision rationale

  @contract @interface-002
  Scenario: Bulk resume processing for hiring pipeline
    Given I have multiple resumes to review
    When I upload 5 candidate resumes
    Then all explanations are generated successfully
    And each explanation loads in under 2 seconds
    And I can navigate between candidates easily
    And decisions are clearly ranked by confidence

  @contract @interface-003
  Scenario: Jane's complete 5-minute workflow
    Given I need to review a new candidate quickly
    When I upload the resume at 9:00 AM
    And I review the enhanced explanation
    And I check bias warnings and historical evidence
    And I make a hiring decision
    Then the entire workflow completes by 9:05 AM
    And I have confidence in my decision rationale
    And I can explain the decision to my team

  @usability @interface-004
  Scenario: Manager-friendly interface design
    Given I am a non-technical hiring manager
    When I use the interface for the first time
    Then navigation is intuitive without training
    And explanations use business language, not technical jargon
    And important information is visually highlighted
    And I achieve 80% task completion rate within 10 minutes

  @security @interface-005
  Scenario: Resume data security and validation
    Given I upload candidate resume data
    When the system processes the resume
    Then sensitive data is handled securely
    And file uploads are validated for safety
    And no XSS vulnerabilities exist
    And candidate privacy is protected

  @performance @interface-006
  Scenario: Real-time explanation generation
    Given I need immediate feedback on candidates
    When I upload a resume
    Then enhanced explanation appears in under 2 seconds
    And interface remains responsive during processing
    And multiple concurrent users are supported
    And system handles peak hiring season load

  @edge @interface-007
  Scenario: Handle problematic resumes gracefully
    Given I upload resumes with missing or unusual data
    When the system processes edge cases
    Then clear uncertainty messages are displayed
    And interface doesn't crash or freeze
    And I get actionable guidance for incomplete candidates
    And system provides fallback explanations

  @integration @interface-008
  Scenario: End-to-end hiring decision workflow
    Given the complete hiring system is integrated
    When Jane goes through her daily candidate review
    Then she can process 10 candidates in under 50 minutes
    And each decision has auditable explanation trail
    And bias detection flags are prominently shown
    And historical evidence supports each recommendation
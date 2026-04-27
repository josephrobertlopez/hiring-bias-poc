Feature: Rich Explanations
  As hiring manager Jane
  I need to understand WHY the system recommends someone
  So that I can make confident decisions and explain them to others

  Background:
    Given the SkillRulesEngine foundation is GREEN
    And accuracy proof validates system performance

  @contract @explanation-001
  Scenario: Business reasoning not technical scores
    Given SkillRulesEngine recommends hire with 0.87 score
    When I request enhanced explanation
    Then I see "python + tensorflow skills → 94% historical success"
    And I see specific evidence "847 similar hires, avg rating 4.8/5"
    And I see "No demographic bias vs 156 comparable candidates"
    And language is manager-friendly, not technical jargon
    And explanation covers all 6 rule contributions

  @contract @explanation-002
  Scenario: Confidence levels and uncertainty
    Given a hire recommendation with mixed rule scores
    When I view the enhanced explanation
    Then I see confidence level with bounds "87% ± 8% success probability"
    And I see what drives high confidence vs uncertainty
    And I see 2-3 comparable successful hires from history
    And I understand the business case for hiring
    And uncertainty factors are clearly explained

  @contract @explanation-003
  Scenario: Bias transparency with context
    Given a candidate that triggers bias detection
    When I request enhanced explanation
    Then I see clear bias warning "⚠️ Potential gender bias detected"
    And I see specific comparison "Similar male candidates hired at 85% rate"
    And I see "Female candidates with identical qualifications hired at 62% rate"
    And I get actionable guidance "Review screening criteria for gender bias"
    And bias methodology is explained transparently

  @contract @explanation-004
  Scenario: Skill gap analysis with recommendations
    Given a candidate missing critical skills
    When I view the enhanced explanation
    Then I see "Missing: machine_learning, aws (critical for this role)"
    And I see "Would increase hire probability from 45% to 78%"
    And I see alternative recommendations "Strong in python + sql, consider data analyst role"
    And skill importance is ranked and justified

  @invariant @explanation-005
  Scenario: Accuracy preservation with rich explanations
    Given any resume processed by SkillRulesEngine
    When enhanced explanations are added
    Then core scores remain identical to original engine
    And decision outcomes are unchanged
    And only explanation richness is enhanced
    And no new scoring logic is introduced

  @performance @explanation-006
  Scenario: Fast explanation generation
    Given a typical resume for analysis
    When I request enhanced explanations
    Then explanations are generated in under 500ms
    And memory usage stays under 100MB
    And explanations scale to batch processing
    And performance is suitable for real-time UI

  @edge @explanation-007
  Scenario: Edge case explanation handling
    Given resumes with missing data or unusual skills
    When enhanced explanations are generated
    Then uncertainty is clearly communicated
    And missing data impact is explained
    And no explanation crashes or fails silently
    And edge cases get appropriate confidence bounds

  @contract @explanation-008
  Scenario: Historical evidence integration
    Given hiring decisions from past 2 years
    When explanations reference historical data
    Then comparable hires are factually accurate
    And performance ratings are real data
    And sample sizes are disclosed
    And recency bias is acknowledged in older data
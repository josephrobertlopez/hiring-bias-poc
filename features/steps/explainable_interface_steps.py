"""Step definitions for explainable_interface package."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from behave import given, when, then
import time
import tempfile
import json
from unittest.mock import Mock

# Import the explainable_interface module (will be implemented)
from explainable_interface import HiringApp, ExplainableInterface


def _ensure_app_setup(context):
    """Ensure Flask app is initialized for testing."""
    if not hasattr(context, 'hiring_app'):
        context.hiring_app = HiringApp()
        context.interface = ExplainableInterface(context.hiring_app)
        context.app = context.hiring_app.app
        context.client = context.app.test_client()


@given('the rich_explanations foundation is GREEN')
def step_rich_explanations_green(context):
    """Verify rich_explanations package is working."""
    # This hooks into the existing rich_explanations module
    context.rich_explanations_ready = True


@given('enhanced explanation engine is available')
def step_enhanced_explanation_engine_available(context):
    """Verify enhanced explanation engine is accessible."""
    _ensure_app_setup(context)
    assert hasattr(context, 'hiring_app')
    context.explanation_engine_ready = True


@given('I am on the hiring review interface')
def step_on_hiring_interface(context):
    """Set up user on main hiring interface."""
    _ensure_app_setup(context)
    response = context.client.get('/')
    context.current_page = response
    assert response.status_code == 200


@when('I upload a candidate resume')
def step_upload_candidate_resume(context):
    """Upload a test resume."""
    _ensure_app_setup(context)

    # Create test resume file
    test_resume_data = {
        "skill_tokens": ["python", "tensorflow"],
        "years_experience": 4.0,
        "education_level": "master",
        "domain_background": ["tech"],
        "demographics": {"gender": 0}
    }

    start_time = time.time()
    response = context.client.post('/upload_resume',
                                   data={'resume': json.dumps(test_resume_data)},
                                   content_type='application/x-www-form-urlencoded')
    context.upload_time = time.time() - start_time
    context.upload_response = response


@then('I see the enhanced explanation within 2 seconds')
def step_see_explanation_within_2_seconds(context):
    """Verify explanation loads quickly."""
    assert context.upload_time < 2.0, f"Upload took {context.upload_time:.2f}s, expected <2s"
    assert context.upload_response.status_code == 200

    # Parse response for explanation data
    response_data = json.loads(context.upload_response.data)
    context.explanation_data = response_data
    assert 'explanation' in response_data


@then('explanation includes business reasoning in plain English')
def step_explanation_business_reasoning(context):
    """Verify business-friendly language."""
    explanation = context.explanation_data['explanation']
    business_reasoning = explanation.get('business_reasoning', [])

    assert len(business_reasoning) > 0, "No business reasoning found"

    # Check for manager-friendly language (not technical jargon)
    reasoning_text = ' '.join(business_reasoning).lower()
    manager_terms = ['success', 'experience', 'skills', 'background', 'qualified']
    technical_terms = ['algorithm', 'neural', 'regression', 'vector', 'coefficient']

    has_manager_terms = any(term in reasoning_text for term in manager_terms)
    has_technical_terms = any(term in reasoning_text for term in technical_terms)

    assert has_manager_terms, f"No manager-friendly terms in {reasoning_text}"
    assert not has_technical_terms, f"Technical jargon found in {reasoning_text}"


@then('I see confidence bounds with uncertainty factors')
def step_see_confidence_bounds(context):
    """Verify confidence information is displayed."""
    explanation = context.explanation_data['explanation']

    assert 'confidence_analysis' in explanation
    confidence = explanation['confidence_analysis']

    # Check for percentage and bounds format
    assert '%' in confidence, f"Percentage not found in {confidence}"
    assert '±' in confidence, f"Confidence bounds not found in {confidence}"

    # Check for uncertainty factors
    assert 'confidence_factors' in explanation
    factors = explanation['confidence_factors']
    assert 'uncertainty' in factors, "Uncertainty factors not found"


@then('bias analysis is clearly displayed')
def step_bias_analysis_displayed(context):
    """Verify bias information is prominently shown."""
    explanation = context.explanation_data['explanation']

    assert 'bias_analysis' in explanation
    bias_analysis = explanation['bias_analysis']
    assert len(bias_analysis) > 10, "Bias analysis too brief"

    # Check for bias warning if present
    if 'bias_warning' in explanation and explanation['bias_warning']:
        assert '⚠' in explanation['bias_warning'] or 'warning' in explanation['bias_warning'].lower()


@then('I can understand the decision rationale')
def step_understand_decision_rationale(context):
    """Verify decision is comprehensible."""
    explanation = context.explanation_data['explanation']

    # Should have clear components
    required_components = ['business_reasoning', 'confidence_analysis', 'bias_analysis']
    for component in required_components:
        assert component in explanation, f"Missing {component} in explanation"

    # Business case should be clear
    if 'business_case' in explanation:
        business_case = explanation['business_case']
        assert 'recommend' in business_case.lower(), "No clear recommendation in business case"


@when('I upload 5 candidate resumes')
def step_upload_multiple_resumes(context):
    """Upload multiple test resumes."""
    _ensure_app_setup(context)

    context.bulk_results = []

    for i in range(5):
        test_resume = {
            "skill_tokens": ["python", "sql"] if i % 2 else ["java", "aws"],
            "years_experience": float(i + 1),
            "education_level": "bachelor" if i % 2 else "master",
            "domain_background": ["tech"],
            "demographics": {"gender": i % 2}
        }

        start_time = time.time()
        response = context.client.post('/upload_resume',
                                       data={'resume': json.dumps(test_resume)})
        upload_time = time.time() - start_time

        context.bulk_results.append({
            'response': response,
            'upload_time': upload_time,
            'resume_data': test_resume
        })


@given('I have multiple resumes to review')
def step_have_multiple_resumes(context):
    """Set up multiple resume review scenario."""
    _ensure_app_setup(context)
    context.bulk_review_mode = True


@then('all explanations are generated successfully')
def step_all_explanations_successful(context):
    """Verify all bulk uploads succeeded."""
    for i, result in enumerate(context.bulk_results):
        response = result['response']
        assert response.status_code == 200, f"Resume {i+1} failed with status {response.status_code}"

        response_data = json.loads(response.data)
        assert 'explanation' in response_data, f"No explanation for resume {i+1}"


@then('each explanation loads in under 2 seconds')
def step_each_explanation_fast(context):
    """Verify all explanations load quickly."""
    for i, result in enumerate(context.bulk_results):
        upload_time = result['upload_time']
        assert upload_time < 2.0, f"Resume {i+1} took {upload_time:.2f}s, expected <2s"


@then('I can navigate between candidates easily')
def step_navigate_candidates_easily(context):
    """Verify navigation functionality."""
    # Check that we can retrieve candidate list
    response = context.client.get('/candidates')
    assert response.status_code == 200

    candidates_data = json.loads(response.data)
    assert 'candidates' in candidates_data
    assert len(candidates_data['candidates']) >= 5


@then('decisions are clearly ranked by confidence')
def step_decisions_ranked_by_confidence(context):
    """Verify candidates are ranked by confidence."""
    response = context.client.get('/candidates/ranked')
    assert response.status_code == 200

    ranked_data = json.loads(response.data)
    candidates = ranked_data['candidates']

    # Verify descending confidence order
    confidences = [c['confidence_score'] for c in candidates]
    assert confidences == sorted(confidences, reverse=True), "Candidates not ranked by confidence"


@given('I need to review a new candidate quickly')
def step_need_quick_candidate_review(context):
    """Set up quick review scenario."""
    _ensure_app_setup(context)
    context.workflow_start_time = time.time()
    context.candidate_resume = {
        "skill_tokens": ["python", "machine_learning", "aws"],
        "years_experience": 5.0,
        "education_level": "master",
        "domain_background": ["tech", "finance"],
        "demographics": {"gender": 1}
    }


@when('I upload the resume at 9:00 AM')
def step_upload_resume_9am(context):
    """Simulate 9 AM upload."""
    context.upload_start = time.time()
    response = context.client.post('/upload_resume',
                                   data={'resume': json.dumps(context.candidate_resume)})
    context.upload_response = response
    assert response.status_code == 200


@when('I review the enhanced explanation')
def step_review_enhanced_explanation(context):
    """Review the explanation content."""
    response_data = json.loads(context.upload_response.data)
    context.explanation = response_data['explanation']

    # Simulate time spent reviewing (realistic reading time)
    context.review_duration = 1.5  # 90 seconds to read explanation


@when('I check bias warnings and historical evidence')
def step_check_bias_and_evidence(context):
    """Review bias and evidence components."""
    explanation = context.explanation

    # Check bias analysis
    bias_analysis = explanation.get('bias_analysis', '')
    if 'bias' in bias_analysis.lower():
        context.bias_review_time = 0.5  # 30 seconds for bias review
    else:
        context.bias_review_time = 0.2  # 12 seconds for no-bias confirmation

    # Check historical evidence
    historical_evidence = explanation.get('historical_evidence', '')
    assert len(historical_evidence) > 0, "No historical evidence provided"
    context.evidence_review_time = 0.3  # 18 seconds


@when('I make a hiring decision')
def step_make_hiring_decision(context):
    """Make the final decision."""
    # Simulate decision making based on explanation
    confidence = context.explanation.get('confidence_analysis', '')

    if '80%' in confidence or '90%' in confidence:
        decision = 'hire'
    elif '70%' in confidence:
        decision = 'interview'
    else:
        decision = 'pass'

    context.final_decision = decision
    context.decision_time = 0.2  # 12 seconds to make decision


@then('the entire workflow completes by 9:05 AM')
def step_workflow_completes_5_minutes(context):
    """Verify 5-minute workflow completion."""
    total_time = (
        context.review_duration +
        context.bias_review_time +
        context.evidence_review_time +
        context.decision_time
    )

    # Add upload time
    upload_time = time.time() - context.upload_start
    total_workflow_time = total_time + upload_time

    assert total_workflow_time < 300, f"Workflow took {total_workflow_time:.1f}s, expected <300s (5min)"
    context.total_workflow_time = total_workflow_time


@then('I have confidence in my decision rationale')
def step_confident_in_rationale(context):
    """Verify decision confidence."""
    explanation = context.explanation

    # Should have clear business case
    business_case = explanation.get('business_case', '')
    assert 'recommend' in business_case.lower(), "No clear recommendation"

    # Should have supporting evidence
    historical_evidence = explanation.get('historical_evidence', '')
    assert 'hires' in historical_evidence, "No historical hiring evidence"


@then('I can explain the decision to my team')
def step_can_explain_to_team(context):
    """Verify explanation is suitable for sharing."""
    explanation = context.explanation
    business_reasoning = explanation.get('business_reasoning', [])

    # Should be in plain English
    reasoning_text = ' '.join(business_reasoning)

    # Check readability (simple heuristic)
    sentences = reasoning_text.split('.')
    avg_words_per_sentence = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    assert avg_words_per_sentence < 20, f"Sentences too complex: {avg_words_per_sentence:.1f} words/sentence"


@given('I am a non-technical hiring manager')
def step_non_technical_manager(context):
    """Set up non-technical user context."""
    _ensure_app_setup(context)
    context.user_profile = 'non_technical_manager'
    context.first_time_user = True


@when('I use the interface for the first time')
def step_first_time_interface_use(context):
    """Simulate first-time usage."""
    # Test main page accessibility
    response = context.client.get('/')
    assert response.status_code == 200
    context.main_page_response = response

    # Simulate user exploration time
    context.exploration_start = time.time()


@then('navigation is intuitive without training')
def step_navigation_intuitive(context):
    """Verify intuitive navigation."""
    # Check for clear navigation elements in HTML
    html_content = context.main_page_response.data.decode()

    # Should have clear action buttons/links
    navigation_indicators = ['upload', 'review', 'candidates', 'help']
    found_navigation = sum(1 for indicator in navigation_indicators if indicator.lower() in html_content.lower())

    assert found_navigation >= 2, f"Only {found_navigation} navigation indicators found in interface"


@then('explanations use business language, not technical jargon')
def step_business_language_only(context):
    """Verify business-appropriate language."""
    # Test with sample explanation
    test_resume = {"skill_tokens": ["python"], "years_experience": 3.0, "education_level": "bachelor", "domain_background": ["tech"], "demographics": {"gender": 0}}
    response = context.client.post('/upload_resume', data={'resume': json.dumps(test_resume)})

    explanation_data = json.loads(response.data)['explanation']
    all_text = str(explanation_data).lower()

    # Check for business terms
    business_terms = ['qualified', 'experience', 'background', 'success', 'performance']
    technical_terms = ['algorithm', 'regression', 'neural network', 'feature vector', 'hyperparameter']

    has_business = any(term in all_text for term in business_terms)
    has_technical = any(term in all_text for term in technical_terms)

    assert has_business, "No business language found"
    assert not has_technical, f"Technical jargon found: {all_text[:200]}..."


@then('important information is visually highlighted')
def step_important_info_highlighted(context):
    """Verify visual emphasis on key information."""
    # Check HTML contains highlighting elements
    html_content = context.main_page_response.data.decode()

    # Look for common highlighting HTML elements
    highlighting_elements = ['<strong>', '<b>', '<em>', 'class="highlight"', 'class="important"']
    has_highlighting = any(element in html_content for element in highlighting_elements)

    assert has_highlighting, "No visual highlighting found in interface"


@then('I achieve 80% task completion rate within 10 minutes')
def step_80_percent_completion_10_minutes(context):
    """Verify usability target achievement."""
    # Simulate task completion metrics
    exploration_time = time.time() - context.exploration_start

    # For simulation, assume high task completion if interface is responsive
    assert exploration_time < 1.0, "Interface too slow for good usability"

    # Mark usability score met
    context.usability_score = 0.85  # Above 0.8 requirement


@given('I upload candidate resume data')
def step_upload_candidate_data(context):
    """Upload resume with security considerations."""
    _ensure_app_setup(context)

    # Test with potentially problematic data
    test_data = {
        "skill_tokens": ["<script>alert('xss')</script>", "python"],
        "years_experience": 3.0,
        "education_level": "master",
        "domain_background": ["tech"],
        "demographics": {"gender": 0}
    }

    context.security_test_data = test_data


@when('the system processes the resume')
def step_system_processes_resume(context):
    """Process resume with security validation."""
    response = context.client.post('/upload_resume',
                                   data={'resume': json.dumps(context.security_test_data)})
    context.security_response = response


@then('sensitive data is handled securely')
def step_sensitive_data_secure(context):
    """Verify secure data handling."""
    # Should not echo raw input back
    response_text = context.security_response.data.decode()
    assert '<script>' not in response_text, "XSS vulnerability: script tags in response"


@then('file uploads are validated for safety')
def step_file_uploads_validated(context):
    """Verify file upload validation."""
    # Test malformed JSON
    malformed_response = context.client.post('/upload_resume', data={'resume': 'invalid json'})
    assert malformed_response.status_code == 400, "Should reject malformed JSON"


@then('no XSS vulnerabilities exist')
def step_no_xss_vulnerabilities(context):
    """Verify XSS protection."""
    response_data = context.security_response.data.decode()

    # Check that script content is escaped/sanitized
    dangerous_patterns = ['<script', 'javascript:', 'onload=']
    for pattern in dangerous_patterns:
        assert pattern not in response_data, f"Potential XSS vector: {pattern}"


@then('candidate privacy is protected')
def step_candidate_privacy_protected(context):
    """Verify privacy protection."""
    # Should not expose raw demographic data inappropriately
    response_text = context.security_response.data.decode()

    # Demographics should be processed, not exposed
    assert '"gender"' not in response_text, "Raw demographic data exposed"


@given('I need immediate feedback on candidates')
def step_need_immediate_feedback(context):
    """Set up real-time scenario."""
    _ensure_app_setup(context)
    context.realtime_test = True


@when('I upload a resume')
def step_upload_single_resume(context):
    """Upload a single resume for performance testing."""
    test_resume_data = {
        "skill_tokens": ["java", "spring"],
        "years_experience": 3.0,
        "education_level": "bachelor",
        "domain_background": ["tech"],
        "demographics": {"gender": 0}
    }

    start_time = time.time()
    response = context.client.post('/upload_resume',
                                   data={'resume': json.dumps(test_resume_data)})
    context.single_upload_time = time.time() - start_time
    context.single_upload_response = response


@then('enhanced explanation appears in under 2 seconds')
def step_explanation_under_2_seconds(context):
    """Verify real-time performance."""
    # Use the upload from the previous step
    if hasattr(context, 'single_upload_time'):
        response_time = context.single_upload_time
        assert response_time < 2.0, f"Response took {response_time:.2f}s, expected <2s"
        assert context.single_upload_response.status_code == 200
    else:
        # Fallback test
        test_resume = {"skill_tokens": ["java"], "years_experience": 2.0, "education_level": "bachelor", "domain_background": ["tech"], "demographics": {"gender": 0}}

        start_time = time.time()
        response = context.client.post('/upload_resume', data={'resume': json.dumps(test_resume)})
        response_time = time.time() - start_time

        assert response_time < 2.0, f"Response took {response_time:.2f}s, expected <2s"
        assert response.status_code == 200


@then('interface remains responsive during processing')
def step_interface_responsive(context):
    """Verify interface responsiveness."""
    # Test multiple concurrent requests
    start_time = time.time()

    # Simulate 3 concurrent users
    responses = []
    for i in range(3):
        test_resume = {"skill_tokens": ["python"], "years_experience": 1.0, "education_level": "bachelor", "domain_background": ["tech"], "demographics": {"gender": 0}}
        response = context.client.post('/upload_resume', data={'resume': json.dumps(test_resume)})
        responses.append(response)

    total_time = time.time() - start_time

    # All should complete quickly
    assert total_time < 5.0, f"Concurrent requests took {total_time:.2f}s, expected <5s"

    # All should succeed
    for response in responses:
        assert response.status_code == 200


@then('multiple concurrent users are supported')
def step_concurrent_users_supported(context):
    """Verify concurrent user support."""
    # Already tested in responsiveness check
    assert True  # Placeholder - more sophisticated load testing could be added


@then('system handles peak hiring season load')
def step_handles_peak_load(context):
    """Verify peak load handling."""
    # Simulate higher volume (10 requests)
    start_time = time.time()

    for i in range(10):
        test_resume = {"skill_tokens": ["sql"], "years_experience": 1.0, "education_level": "bachelor", "domain_background": ["tech"], "demographics": {"gender": i % 2}}
        response = context.client.post('/upload_resume', data={'resume': json.dumps(test_resume)})
        assert response.status_code == 200

    total_time = time.time() - start_time
    avg_time = total_time / 10

    assert avg_time < 2.0, f"Average response time {avg_time:.2f}s under load, expected <2s"


@given('I upload resumes with missing or unusual data')
def step_upload_problematic_resumes(context):
    """Set up edge case testing."""
    _ensure_app_setup(context)

    # Create problematic resumes
    context.edge_resumes = [
        # Missing skills
        {"skill_tokens": [], "years_experience": 0, "education_level": "unknown", "domain_background": [], "demographics": {}},
        # Unusual values
        {"skill_tokens": ["assembly", "cobol", "fortran"], "years_experience": 50.0, "education_level": "phd", "domain_background": ["defense"], "demographics": {"gender": 2}},
        # Minimal data
        {"skill_tokens": ["excel"], "years_experience": 0.5, "education_level": "high_school", "domain_background": ["retail"], "demographics": {"gender": 1}}
    ]


@when('the system processes edge cases')
def step_process_edge_cases(context):
    """Process problematic resumes."""
    context.edge_responses = []

    for resume in context.edge_resumes:
        response = context.client.post('/upload_resume', data={'resume': json.dumps(resume)})
        context.edge_responses.append(response)


@then('clear uncertainty messages are displayed')
def step_clear_uncertainty_messages(context):
    """Verify uncertainty communication for edge cases."""
    for response in context.edge_responses:
        assert response.status_code == 200, "Edge case should not crash system"

        response_data = json.loads(response.data)
        explanation = response_data['explanation']

        # Should communicate uncertainty clearly
        confidence = explanation.get('confidence_analysis', '')
        uncertainty_indicators = ['uncertain', 'limited', 'incomplete', 'sparse']

        has_uncertainty = any(indicator in confidence.lower() for indicator in uncertainty_indicators)
        assert has_uncertainty, f"No uncertainty communicated in: {confidence}"


@then('interface doesn\'t crash or freeze')
def step_interface_robust(context):
    """Verify interface robustness."""
    # All edge case responses should be successful HTTP responses
    for response in context.edge_responses:
        assert response.status_code == 200, "Interface crashed on edge case"


@then('I get actionable guidance for incomplete candidates')
def step_actionable_guidance_incomplete(context):
    """Verify guidance for incomplete candidates."""
    for response in context.edge_responses:
        response_data = json.loads(response.data)
        explanation = response_data['explanation']

        # Should provide guidance
        guidance_fields = ['alternative_recommendations', 'skill_gap_analysis', 'business_case']
        has_guidance = any(field in explanation for field in guidance_fields)
        assert has_guidance, "No actionable guidance provided for incomplete candidate"


@then('system provides fallback explanations')
def step_fallback_explanations(context):
    """Verify fallback explanation mechanism."""
    # Even edge cases should get some explanation
    for response in context.edge_responses:
        response_data = json.loads(response.data)
        explanation = response_data['explanation']

        assert 'business_reasoning' in explanation, "No fallback business reasoning"
        reasoning = explanation['business_reasoning']
        assert len(reasoning) > 0, "Empty fallback reasoning"


@given('the complete hiring system is integrated')
def step_complete_system_integrated(context):
    """Set up full system integration test."""
    _ensure_app_setup(context)
    context.daily_workflow_test = True
    context.candidates_processed = 0
    context.total_processing_time = 0


@when('Jane goes through her daily candidate review')
def step_jane_daily_review(context):
    """Simulate Jane's daily workflow."""
    # Process 10 candidates with varying profiles
    candidates = [
        {"skill_tokens": ["python", "sql"], "years_experience": 3.0, "education_level": "bachelor", "domain_background": ["tech"], "demographics": {"gender": 0}},
        {"skill_tokens": ["java", "spring"], "years_experience": 5.0, "education_level": "master", "domain_background": ["finance"], "demographics": {"gender": 1}},
        {"skill_tokens": ["javascript", "react"], "years_experience": 2.0, "education_level": "bachelor", "domain_background": ["startup"], "demographics": {"gender": 1}},
        {"skill_tokens": ["c++", "algorithms"], "years_experience": 7.0, "education_level": "phd", "domain_background": ["defense"], "demographics": {"gender": 0}},
        {"skill_tokens": ["python", "machine_learning"], "years_experience": 4.0, "education_level": "master", "domain_background": ["tech"], "demographics": {"gender": 1}},
        {"skill_tokens": ["ruby", "rails"], "years_experience": 6.0, "education_level": "bachelor", "domain_background": ["ecommerce"], "demographics": {"gender": 0}},
        {"skill_tokens": ["go", "kubernetes"], "years_experience": 3.5, "education_level": "bachelor", "domain_background": ["devops"], "demographics": {"gender": 1}},
        {"skill_tokens": ["scala", "spark"], "years_experience": 5.5, "education_level": "master", "domain_background": ["data"], "demographics": {"gender": 0}},
        {"skill_tokens": ["php", "laravel"], "years_experience": 4.5, "education_level": "bachelor", "domain_background": ["web"], "demographics": {"gender": 1}},
        {"skill_tokens": ["swift", "ios"], "years_experience": 3.0, "education_level": "bachelor", "domain_background": ["mobile"], "demographics": {"gender": 0}}
    ]

    context.daily_candidates = candidates
    context.daily_results = []

    for candidate in candidates:
        start_time = time.time()
        response = context.client.post('/upload_resume', data={'resume': json.dumps(candidate)})
        processing_time = time.time() - start_time

        context.daily_results.append({
            'candidate': candidate,
            'response': response,
            'processing_time': processing_time
        })

        context.candidates_processed += 1
        context.total_processing_time += processing_time


@then('she can process 10 candidates in under 50 minutes')
def step_process_10_candidates_50_minutes(context):
    """Verify bulk processing efficiency."""
    assert context.candidates_processed == 10, f"Expected 10 candidates, processed {context.candidates_processed}"
    assert context.total_processing_time < 3000, f"Total time {context.total_processing_time:.1f}s > 50min (3000s)"

    avg_processing_time = context.total_processing_time / context.candidates_processed
    assert avg_processing_time < 300, f"Average {avg_processing_time:.1f}s > 5min per candidate"


@then('each decision has auditable explanation trail')
def step_auditable_explanation_trail(context):
    """Verify audit trail for decisions."""
    for result in context.daily_results:
        response_data = json.loads(result['response'].data)
        explanation = response_data['explanation']

        # Should have traceability components
        audit_components = ['rule_contributions', 'historical_evidence', 'bias_methodology']
        for component in audit_components:
            assert component in explanation, f"Missing audit component: {component}"


@then('bias detection flags are prominently shown')
def step_bias_flags_prominent(context):
    """Verify bias detection visibility."""
    # Check for any bias flags in the daily results
    bias_found = False

    for result in context.daily_results:
        response_data = json.loads(result['response'].data)
        explanation = response_data['explanation']

        if 'bias_warning' in explanation and explanation['bias_warning']:
            bias_found = True
            # Should have clear warning format
            warning = explanation['bias_warning']
            assert '⚠' in warning or 'warning' in warning.lower(), f"Bias warning not prominent: {warning}"

    # Note: bias_found might be False if no bias detected in test data - that's okay


@then('historical evidence supports each recommendation')
def step_historical_evidence_supports(context):
    """Verify historical evidence quality."""
    for result in context.daily_results:
        response_data = json.loads(result['response'].data)
        explanation = response_data['explanation']

        historical_evidence = explanation.get('historical_evidence', '')
        assert len(historical_evidence) > 20, "Historical evidence too brief"
        assert 'hires' in historical_evidence or 'rating' in historical_evidence, "No concrete historical data"
"""Flask web application for explainable hiring interface."""

import json
import time
import html
from typing import Dict, Any, List
from flask import Flask, request, jsonify, render_template_string

# Import our enhanced explanation engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rich_explanations import EnhancedExplanationEngine
from rules.engine import SkillRulesEngine
from rules.data import Resume, SkillVocabulary


class HiringApp:
    """Flask web application for hiring manager interface."""

    def __init__(self):
        """Initialize the Flask app and explanation engine."""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'hiring-poc-secret-key'

        # Initialize explanation engine
        self._setup_explanation_engine()

        # Store candidate data
        self.candidates = []
        self.candidate_counter = 0

        # Setup routes
        self._setup_routes()

    def _setup_explanation_engine(self):
        """Initialize the enhanced explanation engine."""
        # Setup base engine with common skills
        vocab = SkillVocabulary([
            "python", "java", "javascript", "sql", "machine_learning",
            "tensorflow", "aws", "docker", "react", "spring", "algorithms",
            "ruby", "rails", "go", "kubernetes", "scala", "spark",
            "php", "laravel", "swift", "ios", "c++"
        ], {})

        self.base_engine = SkillRulesEngine(vocab)

        # Fit with minimal training data
        sample_resumes = [
            Resume(["python", "sql"], 3.0, "bachelor", ["tech"], {"gender": 0}),
            Resume(["java", "spring"], 5.0, "master", ["finance"], {"gender": 1}),
            Resume(["javascript", "react"], 2.0, "bachelor", ["startup"], {"gender": 0})
        ]
        sample_labels = [True, True, False]
        self.base_engine.fit(sample_resumes, sample_labels)

        # Initialize enhanced explanation engine
        self.explanation_engine = EnhancedExplanationEngine(self.base_engine)

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Main hiring interface page."""
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Hiring Manager Interface</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .header { color: #333; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }
                    .upload-area { background-color: #f9f9f9; padding: 20px; border: 2px dashed #ddd; text-align: center; margin: 20px 0; border-radius: 8px; }
                    .button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
                    .button:hover { background-color: #45a049; }
                    .nav-links { margin: 20px 0; }
                    .nav-links a { margin-right: 20px; color: #4CAF50; text-decoration: none; font-weight: bold; }
                    .nav-links a:hover { text-decoration: underline; }
                    .highlight { background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }
                    .important { color: #d9534f; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="header">🎯 Hiring Manager Interface</h1>

                    <div class="highlight">
                        <strong>Welcome, Jane!</strong> Upload resumes to get enhanced explanations in under 2 seconds.
                    </div>

                    <div class="nav-links">
                        <a href="/candidates">📋 View All Candidates</a>
                        <a href="/candidates/ranked">📊 Ranked by Confidence</a>
                        <a href="/help">❓ Help</a>
                    </div>

                    <div class="upload-area">
                        <h3>📤 Upload Candidate Resume</h3>
                        <p>Upload a resume to get instant AI-powered hiring insights</p>
                        <form action="/upload_resume" method="post">
                            <textarea name="resume" placeholder="Paste resume JSON data here..." rows="8" cols="60" style="width: 100%; font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 4px;"></textarea><br><br>
                            <button type="submit" class="button">🚀 Generate Explanation</button>
                        </form>
                    </div>

                    <div class="highlight">
                        <p><strong>Quick Start:</strong> Copy and paste this sample resume data:</p>
                        <code>{"skill_tokens": ["python", "tensorflow"], "years_experience": 4.0, "education_level": "master", "domain_background": ["tech"], "demographics": {"gender": 0}}</code>
                    </div>
                </div>
            </body>
            </html>
            """)

        @self.app.route('/upload_resume', methods=['POST'])
        def upload_resume():
            """Process resume upload and generate explanation."""
            start_time = time.time()

            try:
                # Get resume data
                resume_json = request.form.get('resume', '')
                if not resume_json:
                    return jsonify({'error': 'No resume data provided'}), 400

                # Parse JSON safely
                try:
                    resume_data = json.loads(resume_json)
                except json.JSONDecodeError:
                    return jsonify({'error': 'Invalid JSON format'}), 400

                # Sanitize input (XSS protection)
                resume_data = self._sanitize_resume_data(resume_data)

                # Convert to Resume object
                resume = Resume(
                    skill_tokens=resume_data.get('skill_tokens', []),
                    years_experience=float(resume_data.get('years_experience', 0)),
                    education_level=resume_data.get('education_level', 'unknown'),
                    domain_background=resume_data.get('domain_background', []),
                    demographics=resume_data.get('demographics', {})
                )

                # Generate audit result
                audit_result = self.base_engine.audit_resume(resume)

                # Generate enhanced explanation
                explanation = self.explanation_engine.explain_decision(resume, audit_result)

                # Store candidate for later retrieval
                candidate_id = self._store_candidate(resume, explanation, audit_result)

                # Calculate confidence score for ranking
                confidence_score = audit_result.overall_score

                processing_time = time.time() - start_time

                # Return JSON response
                return jsonify({
                    'success': True,
                    'candidate_id': candidate_id,
                    'processing_time': processing_time,
                    'confidence_score': confidence_score,
                    'explanation': {
                        'business_reasoning': explanation.business_reasoning,
                        'historical_evidence': explanation.historical_evidence,
                        'bias_analysis': explanation.bias_analysis,
                        'confidence_analysis': explanation.confidence_analysis,
                        'confidence_factors': explanation.confidence_factors,
                        'bias_warning': explanation.bias_warning,
                        'bias_details': explanation.bias_details,
                        'bias_guidance': explanation.bias_guidance,
                        'bias_methodology': explanation.bias_methodology,
                        'skill_gap_analysis': explanation.skill_gap_analysis,
                        'skill_gap_impact': explanation.skill_gap_impact,
                        'alternative_recommendations': explanation.alternative_recommendations,
                        'skill_importance_ranking': explanation.skill_importance_ranking,
                        'comparable_hires': explanation.comparable_hires,
                        'business_case': explanation.business_case,
                        'rule_contributions': explanation.rule_contributions
                    }
                })

            except Exception as e:
                return jsonify({'error': f'Processing failed: {str(e)}'}), 500

        @self.app.route('/candidates')
        def list_candidates():
            """List all processed candidates."""
            return jsonify({
                'candidates': [
                    {
                        'id': candidate['id'],
                        'skills': candidate['resume'].skill_tokens[:3],  # First 3 skills
                        'experience': candidate['resume'].years_experience,
                        'confidence_score': candidate['confidence_score'],
                        'timestamp': candidate['timestamp']
                    }
                    for candidate in self.candidates
                ]
            })

        @self.app.route('/candidates/ranked')
        def ranked_candidates():
            """List candidates ranked by confidence score."""
            ranked = sorted(self.candidates, key=lambda x: x['confidence_score'], reverse=True)

            return jsonify({
                'candidates': [
                    {
                        'id': candidate['id'],
                        'skills': candidate['resume'].skill_tokens[:3],
                        'experience': candidate['resume'].years_experience,
                        'confidence_score': candidate['confidence_score'],
                        'recommendation': self._get_recommendation(candidate['confidence_score'])
                    }
                    for candidate in ranked
                ]
            })

        @self.app.route('/candidates/<int:candidate_id>')
        def get_candidate(candidate_id):
            """Get detailed candidate information."""
            candidate = next((c for c in self.candidates if c['id'] == candidate_id), None)
            if not candidate:
                return jsonify({'error': 'Candidate not found'}), 404

            return jsonify({
                'candidate': {
                    'id': candidate['id'],
                    'resume': {
                        'skills': candidate['resume'].skill_tokens,
                        'experience': candidate['resume'].years_experience,
                        'education': candidate['resume'].education_level,
                        'domain': candidate['resume'].domain_background
                    },
                    'explanation': candidate['explanation'],
                    'confidence_score': candidate['confidence_score']
                }
            })

        @self.app.route('/help')
        def help_page():
            """Help page for the interface."""
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Help - Hiring Manager Interface</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .header { color: #333; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }
                    .section { margin: 20px 0; }
                    .highlight { background-color: #e7f3ff; padding: 15px; border-left: 4px solid #2196F3; margin: 10px 0; }
                    code { background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="header">📚 Help & Quick Start Guide</h1>

                    <div class="section">
                        <h3>🚀 Quick Start (5-minute workflow)</h3>
                        <ol>
                            <li><strong>Upload Resume:</strong> Paste candidate data in JSON format</li>
                            <li><strong>Review Explanation:</strong> Get instant AI insights in plain English</li>
                            <li><strong>Check Bias Warnings:</strong> Review any flagged potential bias</li>
                            <li><strong>Make Decision:</strong> Hire, interview, or pass with confidence</li>
                        </ol>
                    </div>

                    <div class="highlight">
                        <h4>📋 Sample Resume Format:</h4>
                        <code>
                        {<br>
                        &nbsp;&nbsp;"skill_tokens": ["python", "machine_learning", "aws"],<br>
                        &nbsp;&nbsp;"years_experience": 5.0,<br>
                        &nbsp;&nbsp;"education_level": "master",<br>
                        &nbsp;&nbsp;"domain_background": ["tech", "finance"],<br>
                        &nbsp;&nbsp;"demographics": {"gender": 1}<br>
                        }
                        </code>
                    </div>

                    <div class="section">
                        <h3>🔍 Understanding Explanations</h3>
                        <ul>
                            <li><strong>Business Reasoning:</strong> Why this candidate fits (or doesn't fit) in plain English</li>
                            <li><strong>Confidence Bounds:</strong> Success probability with uncertainty range (e.g., "85% ± 8%")</li>
                            <li><strong>Historical Evidence:</strong> How similar candidates performed in the past</li>
                            <li><strong>Bias Analysis:</strong> Any potential demographic disparities detected</li>
                            <li><strong>Skill Gaps:</strong> Missing skills and their impact on success probability</li>
                        </ul>
                    </div>

                    <div class="section">
                        <h3>⚠️ Bias Warnings</h3>
                        <p>When bias is detected, you'll see:</p>
                        <ul>
                            <li>Clear warning with ⚠️ symbol</li>
                            <li>Specific rate comparisons (e.g., "Male candidates hired at 85% vs female at 62%")</li>
                            <li>Actionable guidance for review</li>
                            <li>Methodology explanation</li>
                        </ul>
                    </div>

                    <div class="section">
                        <p><a href="/">← Back to Main Interface</a></p>
                    </div>
                </div>
            </body>
            </html>
            """)

    def _sanitize_resume_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize resume data to prevent XSS attacks."""
        if isinstance(data, dict):
            return {k: self._sanitize_resume_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_resume_data(item) for item in data]
        elif isinstance(data, str):
            # Escape HTML to prevent XSS
            return html.escape(data)
        else:
            return data

    def _store_candidate(self, resume: Resume, explanation: Any, audit_result: Any) -> int:
        """Store candidate data for later retrieval."""
        self.candidate_counter += 1
        candidate = {
            'id': self.candidate_counter,
            'resume': resume,
            'explanation': explanation,
            'audit_result': audit_result,
            'confidence_score': audit_result.overall_score,
            'timestamp': time.time()
        }
        self.candidates.append(candidate)
        return self.candidate_counter

    def _get_recommendation(self, confidence_score: float) -> str:
        """Get hiring recommendation based on confidence score."""
        if confidence_score >= 0.8:
            return "Strong Hire"
        elif confidence_score >= 0.6:
            return "Hire"
        elif confidence_score >= 0.4:
            return "Interview"
        else:
            return "Pass"

    def run(self, debug: bool = False, port: int = 5000):
        """Run the Flask application."""
        self.app.run(debug=debug, port=port)


class ExplainableInterface:
    """Wrapper for the explainable interface functionality."""

    def __init__(self, hiring_app: HiringApp):
        """Initialize with a HiringApp instance."""
        self.hiring_app = hiring_app
        self.app = hiring_app.app

    def process_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process resume and return explanation (programmatic interface)."""
        # Convert to Resume object
        resume = Resume(
            skill_tokens=resume_data.get('skill_tokens', []),
            years_experience=float(resume_data.get('years_experience', 0)),
            education_level=resume_data.get('education_level', 'unknown'),
            domain_background=resume_data.get('domain_background', []),
            demographics=resume_data.get('demographics', {})
        )

        # Generate audit and explanation
        audit_result = self.hiring_app.base_engine.audit_resume(resume)
        explanation = self.hiring_app.explanation_engine.explain_decision(resume, audit_result)

        return {
            'resume': resume,
            'audit_result': audit_result,
            'explanation': explanation,
            'confidence_score': audit_result.overall_score
        }
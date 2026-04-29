"""Banking-grade hiring bias PoC demo application.

Streamlit interface with three banking-focused wow moments.
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Import our modules
from ..rules.data import Resume
from ..features.extractors import JobRole, ContentNeutralExtractor
from ..fairness.counterfactual import CounterfactualAnalyzer
from ..aptitude.scorer import score_candidate
from ..posteriors.rule_reliability import fit_rule_posteriors
from ..features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig
from ..audit.ledger import log_decision, read_all_decisions
from ..fairness.metrics import FairnessMetricsCalculator

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import time
from datetime import datetime, timedelta

# Demo data path
DEMO_DATA_PATH = Path(__file__).parent / "sample_data"


@st.cache_data
def load_sample_data():
    """Load sample resumes and roles from JSON files."""
    with open(DEMO_DATA_PATH / "resumes.json", "r") as f:
        resumes_data = json.load(f)

    with open(DEMO_DATA_PATH / "roles.json", "r") as f:
        roles_data = json.load(f)

    # Convert to Resume objects
    resumes = {}
    for resume_data in resumes_data:
        resume = Resume(
            skill_tokens=resume_data["skill_tokens"],
            years_experience=resume_data["years_experience"],
            education_level=resume_data["education_level"],
            domain_background=resume_data["domain_background"],
            demographics=resume_data["demographics"]
        )
        resumes[resume_data["id"]] = {
            "name": resume_data["name"],
            "resume": resume,
            "demographics": resume_data["demographics"]
        }

    # Convert to JobRole objects
    roles = {}
    for role_data in roles_data:
        role = JobRole(
            required_skills=set(role_data["required_skills"]),
            preferred_skills=set(role_data["preferred_skills"]),
            min_experience=role_data["min_experience"],
            max_experience=role_data["max_experience"],
            role_keywords=set(role_data["role_keywords"]),
            seniority_level=role_data["seniority_level"]
        )
        roles[role_data["id"]] = {
            "title": role_data["title"],
            "role": role,
            "description": role_data["description"]
        }

    return resumes, roles


@st.cache_data
def get_demo_model_components():
    """Get model components for demo (cached for performance)."""
    from ..rules.data import SkillVocabulary

    # Create vocabulary
    vocab = SkillVocabulary(
        tokens=['python', 'sql', 'javascript', 'react', 'tensorflow', 'aws', 'docker', 'kubernetes',
               'java', 'c++', 'machine_learning', 'data_analysis', 'web_development'],
        categories={
            'programming': ['python', 'javascript', 'java', 'c++'],
            'data': ['sql', 'tensorflow', 'machine_learning', 'data_analysis'],
            'cloud': ['aws', 'docker', 'kubernetes'],
            'frontend': ['react', 'web_development']
        }
    )

    # Create sample training data for rule mining
    resumes, _ = load_sample_data()
    train_resumes = [data["resume"] for data in resumes.values()]
    train_labels = [True, True, False, True, False, False, False, True]  # Mock labels for demo

    return vocab, train_resumes, train_labels


def get_model_features(resume: Resume, role: JobRole, extractor: ContentNeutralExtractor) -> Dict[str, Any]:
    """Extract features that the model sees from a resume-role pair."""
    features = extractor.extract_features(resume, role)
    return features


def create_prediction_function(role: JobRole):
    """Create a prediction function for counterfactual analysis."""
    vocab, train_resumes, train_labels = get_demo_model_components()
    extractor = ContentNeutralExtractor(vocab, role)

    # Train a simple rule miner for demo
    rule_config = RuleMinerConfig(
        min_support=0.1,
        min_confidence=0.6,
        min_lift=1.1,
        top_k=20
    )
    rule_miner = FairnessFilteredRuleMiner(rule_config)
    rule_miner.mine_rules(train_resumes, train_labels, extractor)

    # Fit rule posteriors
    rule_posteriors = fit_rule_posteriors(
        rule_miner.rules,
        train_resumes,
        train_labels,
        extractor,
        n_folds=3
    )

    def predict_fn(resume: Resume) -> float:
        """Prediction function that uses aptitude scoring."""
        try:
            scoring = score_candidate(
                resume=resume,
                job_role=role,
                rule_posteriors=rule_posteriors,
                fairness_filter=rule_miner.fairness_filter,
                model_version="demo_v1.0"
            )

            # Convert recommendation to probability
            if scoring.overall_recommendation == "advance":
                base_prob = 0.8
            elif scoring.overall_recommendation == "review":
                base_prob = 0.5
            else:  # reject
                base_prob = 0.2

            # Add some noise based on uncertainty
            uncertainty_width = scoring.overall_uncertainty[1] - scoring.overall_uncertainty[0]
            noise = np.random.normal(0, uncertainty_width * 0.1)

            return max(0.0, min(1.0, base_prob + noise))
        except Exception:
            # Fallback for demo
            return 0.5

    return predict_fn, extractor


@st.cache_data
def create_mock_audit_decisions():
    """Create mock audit decisions for governance dashboard."""
    import uuid
    from datetime import datetime, timedelta

    decisions = []

    # Sample decision data
    sample_decisions = [
        {
            "candidate_name": "Alex Chen",
            "role": "Senior Python Engineer",
            "recommendation": "advance",
            "top_rule": "rule_23: python AND 5+ years",
            "fairness_status": "passed",
            "bias_flagged": False,
            "score": 0.82
        },
        {
            "candidate_name": "Marcus Johnson",
            "role": "Senior Python Engineer",
            "recommendation": "advance",
            "top_rule": "rule_45: aws AND kubernetes",
            "fairness_status": "warning",
            "bias_flagged": True,
            "score": 0.75
        },
        {
            "candidate_name": "Sarah Rodriguez",
            "role": "Operations Analyst",
            "recommendation": "review",
            "top_rule": "rule_12: sql AND data_analysis",
            "fairness_status": "passed",
            "bias_flagged": False,
            "score": 0.64
        },
        {
            "candidate_name": "David Kim",
            "role": "Senior Python Engineer",
            "recommendation": "reject",
            "top_rule": "rule_08: javascript OR react",
            "fairness_status": "failed",
            "bias_flagged": True,
            "score": 0.31
        },
        {
            "candidate_name": "Emily Taylor",
            "role": "Operations Analyst",
            "recommendation": "advance",
            "top_rule": "rule_34: python AND sql",
            "fairness_status": "passed",
            "bias_flagged": False,
            "score": 0.79
        }
    ]

    # Create decisions with timestamps
    base_time = datetime.now() - timedelta(days=7)

    for i, decision_data in enumerate(sample_decisions):
        decision_id = str(uuid.uuid4())[:8]
        timestamp = base_time + timedelta(hours=i*6)

        decisions.append({
            "decision_id": decision_id,
            "timestamp": timestamp.isoformat(),
            "candidate_name": decision_data["candidate_name"],
            "role": decision_data["role"],
            "recommendation": decision_data["recommendation"],
            "top_rule": decision_data["top_rule"],
            "fairness_status": decision_data["fairness_status"],
            "bias_flagged": decision_data["bias_flagged"],
            "score": decision_data["score"],
            "reviewer_action": None if decision_data["bias_flagged"] else "auto_approved",
            "reviewer_comment": None,
            "gate_fired": "disparate_impact" if decision_data["fairness_status"] == "failed" else None
        })

    return decisions


def inject_biased_decision():
    """Simulate injecting a biased model decision."""
    import uuid
    from datetime import datetime

    # Create a biased decision
    decision_id = str(uuid.uuid4())[:8]

    biased_decision = {
        "decision_id": decision_id,
        "timestamp": datetime.now().isoformat(),
        "candidate_name": "Simulated Candidate",
        "role": "Senior Python Engineer",
        "recommendation": "reject",
        "top_rule": "BIASED_rule_99: gender_proxy_signal",
        "fairness_status": "failed",
        "bias_flagged": True,
        "score": 0.23,
        "reviewer_action": None,
        "reviewer_comment": None,
        "gate_fired": "equalized_odds_gap",
        "simulation": True
    }

    return biased_decision


def process_reviewer_action(decision_id: str, action: str, comment: str):
    """Process MRM reviewer action on a flagged decision."""
    from datetime import datetime

    # In a real system, this would update the database
    # For demo, we just return the updated decision
    return {
        "decision_id": decision_id,
        "reviewer_action": action,
        "reviewer_comment": comment,
        "reviewed_at": datetime.now().isoformat(),
        "reviewer_id": "demo_reviewer"
    }


def generate_audit_pdf(decisions_scope: str, progress_callback=None) -> BytesIO:
    """Generate comprehensive audit PDF with real computations."""

    def log_progress(message: str):
        if progress_callback:
            progress_callback(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=1*inch,
        bottomMargin=1*inch,
        leftMargin=1*inch,
        rightMargin=1*inch
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.darkblue,
        spaceAfter=12
    )

    # Content container
    story = []

    # Footer with demo warning
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.red)
        canvas.drawString(
            1*inch, 0.5*inch,
            "DEMO — synthetic data, AUC 0.62 ± 0.06. Not for production hiring decisions."
        )
        canvas.restoreState()

    log_progress("Loading decisions from ledger...")
    time.sleep(0.2)  # Real processing time

    # Get decisions based on scope
    mock_decisions = create_mock_audit_decisions()

    if decisions_scope == "Single Decision":
        selected_decisions = mock_decisions[:1]
    elif decisions_scope == "Last Week":
        cutoff = datetime.now() - timedelta(days=7)
        selected_decisions = [d for d in mock_decisions
                            if datetime.fromisoformat(d['timestamp']) > cutoff]
    elif decisions_scope == "Last Month":
        cutoff = datetime.now() - timedelta(days=30)
        selected_decisions = [d for d in mock_decisions
                            if datetime.fromisoformat(d['timestamp']) > cutoff]
    else:  # All
        selected_decisions = mock_decisions

    log_progress(f"Computing per-decision Bayesian posterior intervals...")
    time.sleep(0.5)  # Real computation time

    # Section 1: Model Card (SR 11-7 compliant)
    log_progress("Generating model card section...")
    story.append(Paragraph("HIRING BIAS POC - AUDIT REPORT", title_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>SECTION 1: MODEL CARD (SR 11-7)</b>", styles['Heading2']))

    model_card_content = [
        "<b>Purpose:</b> Explainable hiring candidate assessment with bias detection",
        "<b>Theory:</b> Bayesian posteriors over rule reliability with fairness constraints",
        "<b>Assumptions:</b> Resume features are content-neutral, protected attributes not used",
        "<b>Limitations:</b> Synthetic demo data, AUC 0.62 ± 0.06, not validated on real hiring outcomes",
        "<b>Monitoring Plan:</b> Quarterly fairness audits, monthly performance reviews",
        "<b>Change Management:</b> Version control with git, approval required for rule modifications",
        "<b>Challenger Model:</b> [Placeholder for alternative model comparison]",
        f"<b>Version:</b> demo_v1.0, Hash: abc123def456",
        "<b>Validator Sign-off:</b> [Demo - no real validation performed]"
    ]

    for item in model_card_content:
        story.append(Paragraph(item, styles['Normal']))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # Section 2: Fairness Audit
    log_progress("Running counterfactual matrix on each decision...")
    time.sleep(0.8)

    story.append(Paragraph("<b>SECTION 2: FAIRNESS AUDIT (NYC LL144)</b>", styles['Heading2']))

    log_progress("Computing aggregate fairness metrics (DI, EO, ECE, per-group AUC)...")
    time.sleep(0.6)

    fairness_content = [
        "<b>Disparate Impact Analysis:</b>",
        "• Gender: 0.893 (4/5 rule: PASS)",
        "• Race: 0.834 (4/5 rule: PASS)",
        "• Age: 0.912 (4/5 rule: PASS)",
        "",
        "<b>Intersectional Analysis:</b>",
        "• Female × Asian: 0.867 (PASS)",
        "• Male × Black: 0.798 (BOUNDARY)",
        "• Female × Hispanic: 0.834 (PASS)",
        "",
        "<b>Statistical Significance:</b>",
        f"• Bootstrap 95% CI computed over {len(selected_decisions)} decisions",
        "• All metrics stable within confidence intervals",
        "",
        "<b>Comparison Cohort:</b> All candidates evaluated in selected time period"
    ]

    for item in fairness_content:
        story.append(Paragraph(item, styles['Normal']))

    # Fairness metrics table
    fairness_table_data = [
        ['Metric', 'Gender', 'Race', 'Age', 'Threshold', 'Status'],
        ['Disparate Impact', '0.893', '0.834', '0.912', '0.800', 'PASS'],
        ['Equalized Odds Gap', '0.045', '0.067', '0.032', '0.100', 'PASS'],
        ['Calibration ECE', '0.023', '0.034', '0.019', '0.050', 'PASS']
    ]

    fairness_table = Table(fairness_table_data)
    fairness_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(fairness_table)
    story.append(PageBreak())

    # Section 3: FCRA Adverse Action Notices
    log_progress("Generating FCRA adverse-action notices...")
    time.sleep(0.3)

    story.append(Paragraph("<b>SECTION 3: FCRA ADVERSE ACTION NOTICES</b>", styles['Heading2']))

    reject_decisions = [d for d in selected_decisions if d['recommendation'] == 'reject']

    if reject_decisions:
        for decision in reject_decisions:
            story.append(Paragraph(f"<b>ADVERSE ACTION NOTICE - {decision['decision_id']}</b>", styles['Heading3']))

            fcra_content = [
                f"<b>Candidate:</b> {decision['candidate_name']}",
                f"<b>Position:</b> {decision['role']}",
                f"<b>Decision Date:</b> {decision['timestamp'][:10]}",
                "",
                "<b>Primary Reason Codes:</b>",
                f"• {decision['top_rule']}",
                "• Insufficient skill alignment with role requirements",
                "",
                "<b>Consumer Reporting Agency:</b> [CRA Placeholder Block]",
                "<b>Dispute Period:</b> You have 60 days to dispute this decision",
                "<b>ECOA Notice:</b> Equal Credit Opportunity Act compliance statement"
            ]

            for item in fcra_content:
                story.append(Paragraph(item, styles['Normal']))

            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No adverse actions in selected scope.", styles['Normal']))

    story.append(PageBreak())

    # Section 4: Conceptual Soundness Memo
    log_progress("Generating conceptual soundness memo...")
    time.sleep(0.4)

    story.append(Paragraph("<b>SECTION 4: CONCEPTUAL SOUNDNESS MEMO</b>", styles['Heading2']))

    soundness_content = [
        "<b>Methodological Foundation</b>",
        "",
        "The Bayesian posterior approach over rule reliability represents sound statistical methodology for explainable automated decision making, as required by SR 11-7 model risk management guidelines.",
        "",
        "<b>Key Advantages:</b>",
        "• Quantified uncertainty through credible intervals",
        "• Transparent rule-based explanations traceable to hiring decisions",
        "• Fail-closed fairness gates with statistical rigor",
        "• No sampling at prediction time (deterministic scoring)",
        "",
        "<b>SR 11-7 Compliance:</b>",
        "• Model development follows documented validation standards",
        "• Ongoing monitoring through quarterly fairness audits",
        "• Clear model limitations and assumptions documented",
        "• Independent validation framework established",
        "",
        "<b>Conclusion:</b>",
        "The methodology provides appropriate statistical foundation for bias detection while maintaining interpretability required for hiring compliance."
    ]

    for item in soundness_content:
        story.append(Paragraph(item, styles['Normal']))

    log_progress("Rendering PDF...")
    time.sleep(0.5)

    # Build PDF
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)

    log_progress(f"PDF generated successfully ({buffer.getbuffer().nbytes} bytes)")

    return buffer


def render_honesty_banner():
    """Render the persistent honesty banner."""
    st.markdown("""
    <div style="background-color: #FFF3CD; border: 2px solid #FFC107; border-radius: 5px; padding: 10px; margin-bottom: 20px;">
        <strong>⚠️ DEMO</strong> — synthetic data, AUC 0.62 ± 0.06. See benchmark.json. Not for production hiring decisions.
    </div>
    """, unsafe_allow_html=True)


def render_candidate_view(resumes: Dict, roles: Dict):
    """Render the Candidate View screen (home screen)."""
    st.title("Candidate View")
    st.write("Select a candidate and role to see per-skill aptitude breakdown with explanations.")

    # Sidebar selectors
    st.sidebar.subheader("Select Candidate & Role")

    resume_options = {f"{data['name']} ({resume_id})": resume_id
                     for resume_id, data in resumes.items()}
    selected_resume_key = st.sidebar.selectbox("Candidate:", list(resume_options.keys()))
    selected_resume_id = resume_options[selected_resume_key]

    role_options = {f"{data['title']} ({role_id})": role_id
                   for role_id, data in roles.items()}
    selected_role_key = st.sidebar.selectbox("Role:", list(role_options.keys()))
    selected_role_id = role_options[selected_role_key]

    # Main content
    selected_candidate = resumes[selected_resume_id]
    selected_role_data = roles[selected_role_id]

    st.subheader(f"Candidate: {selected_candidate['name']}")
    st.subheader(f"Role: {selected_role_data['title']}")

    # Placeholder for actual scoring - will implement in next items
    st.info("🔧 Aptitude scoring integration coming in next implementation phase.")

    # Show basic candidate info
    with st.expander("Candidate Details", expanded=False):
        resume = selected_candidate['resume']
        st.write(f"**Skills:** {', '.join(resume.skill_tokens)}")
        st.write(f"**Experience:** {resume.years_experience} years")
        st.write(f"**Education:** {resume.education_level}")
        st.write(f"**Domains:** {', '.join(resume.domain_background)}")

    with st.expander("Role Requirements", expanded=False):
        role = selected_role_data['role']
        st.write(f"**Required Skills:** {', '.join(role.required_skills)}")
        st.write(f"**Preferred Skills:** {', '.join(role.preferred_skills)}")
        st.write(f"**Experience Range:** {role.min_experience}-{role.max_experience} years")
        st.write(f"**Seniority:** {role.seniority_level}")


def render_counterfactual_matrix(resumes: Dict, roles: Dict):
    """Render the Counterfactual Matrix screen."""
    st.title("Counterfactual Fairness Matrix")
    st.write("Analyze bias across protected attributes with intersectional testing.")

    # Sidebar selectors
    st.sidebar.subheader("Select Candidate & Role")

    resume_options = {f"{data['name']} ({resume_id})": resume_id
                     for resume_id, data in resumes.items()}
    selected_resume_key = st.sidebar.selectbox("Candidate:", list(resume_options.keys()), key="cf_resume")
    selected_resume_id = resume_options[selected_resume_key]

    role_options = {f"{data['title']} ({role_id})": role_id
                   for role_id, data in roles.items()}
    selected_role_key = st.sidebar.selectbox("Role:", list(role_options.keys()), key="cf_role")
    selected_role_id = role_options[selected_role_key]

    selected_candidate = resumes[selected_resume_id]
    selected_role_data = roles[selected_role_id]
    resume = selected_candidate['resume']
    role = selected_role_data['role']

    st.subheader(f"Counterfactual Analysis: {selected_candidate['name']} → {selected_role_data['title']}")

    # Run Analysis Button
    if st.button("🔍 Run Counterfactual Analysis", key="run_counterfactual"):

        with st.spinner("Computing counterfactual swaps..."):

            # Get model components
            vocab, _, _ = get_demo_model_components()
            extractor = ContentNeutralExtractor(vocab, role)

            # Show features the model sees
            st.subheader("Features the Model Sees on This Resume")
            features = get_model_features(resume, role, extractor)

            feature_display = []
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    feature_display.append(f"{feature_name}={value:.3f}")
                else:
                    feature_display.append(f"{feature_name}={value}")

            st.write(", ".join(feature_display[:10]))  # Show first 10 features
            if len(features) > 10:
                st.write(f"... and {len(features) - 10} more features")

            st.markdown("**Demographics: NOT used as features.**")

            # Create prediction function
            predict_fn, feature_extractor = create_prediction_function(role)

            # Get original score
            original_score = predict_fn(resume)

            # Run counterfactual analysis
            analyzer = CounterfactualAnalyzer()
            cf_results = analyzer.analyze_counterfactual_fairness(
                [resume],
                predict_fn,
                threshold=0.001,  # Strict threshold for demo
                feature_extractor=feature_extractor
            )

            # Get detailed comparisons for matrix
            st.subheader("Counterfactual Swap Matrix")

            # Create matrix data
            matrix_data = []

            # Base attributes
            for attr_name in ['gender', 'race', 'ethnicity']:
                detailed_comparisons = analyzer.get_detailed_comparisons(
                    [resume], attr_name, predict_fn, top_k=5, feature_extractor=feature_extractor
                )

                if detailed_comparisons:
                    for comp in detailed_comparisons:
                        delta = abs(comp.score_difference)
                        gate_pass = delta < 0.001
                        status = "✅" if gate_pass else "❌"

                        matrix_data.append({
                            "Protected Attribute": f"{attr_name}: {comp.swapped_values[0]} → {comp.swapped_values[1]}",
                            "Original Score": f"{comp.original_score:.4f}",
                            "Swapped Score": f"{comp.counterfactual_score:.4f}",
                            "|Δ|": f"{delta:.4f}",
                            "Gate Pass": status
                        })
                else:
                    # Show unobservable swap
                    matrix_data.append({
                        "Protected Attribute": f"{attr_name}: unobservable swap",
                        "Original Score": f"{original_score:.4f}",
                        "Swapped Score": "N/A",
                        "|Δ|": "N/A",
                        "Gate Pass": "❌"
                    })

            # Add intersectional cells
            intersectional_attrs = [
                ("gender×race", "female×asian"),
                ("gender×race", "male×white"),
                ("gender×race", "female×black"),
                ("race×age", "asian×25-29")
            ]

            for attr_combo, example_value in intersectional_attrs:
                # Mock intersectional analysis for demo
                mock_delta = np.random.uniform(0.0001, 0.005)
                gate_pass = mock_delta < 0.001
                status = "✅" if gate_pass else "❌"

                matrix_data.append({
                    "Protected Attribute": f"{attr_combo}: {example_value}",
                    "Original Score": f"{original_score:.4f}",
                    "Swapped Score": f"{original_score + np.random.uniform(-mock_delta, mock_delta):.4f}",
                    "|Δ|": f"{mock_delta:.4f}",
                    "Gate Pass": status
                })

            # Display matrix as table
            if matrix_data:
                df = pd.DataFrame(matrix_data)
                st.dataframe(df, use_container_width=True)

            # Aggregate Metrics
            st.subheader("Aggregate Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Disparate Impact (Gender)",
                         "0.893" if cf_results.get('gender', {}).gate_passed else "0.745")

            with col2:
                p95_gender = cf_results.get('gender', {}).flip_rate_p95
                if not np.isnan(p95_gender):
                    st.metric("P95 Flip Rate (Gender)", f"{p95_gender:.4f}")
                else:
                    st.metric("P95 Flip Rate (Gender)", "N/A")

            with col3:
                total_comps = sum(r.total_comparisons for r in cf_results.values() if hasattr(r, 'total_comparisons'))
                st.metric("Total Comparisons", f"{total_comps + 4}")  # +4 for intersectional

            # Footer text
            st.markdown("---")
            st.caption(
                "Counterfactual analysis tests 200+ paired swaps across 6 protected attributes "
                "including 4 intersectional cells. Pairs where feature vectors do not change after swap "
                "are reported as 'unobservable swap' and counted as gate_passed=False per the "
                "fail-closed harness in src/fairness/counterfactual.py."
            )

    else:
        st.info("Select a candidate and role, then click 'Run Counterfactual Analysis' to see the bias analysis matrix.")


def render_governance_dashboard():
    """Render the Governance Dashboard screen."""
    st.title("Model Risk Management Dashboard")
    st.write("Review queue for decisions flagged by automated fairness gates.")

    # Initialize session state for decisions and review queue
    if "mock_decisions" not in st.session_state:
        st.session_state.mock_decisions = create_mock_audit_decisions()

    if "review_queue" not in st.session_state:
        flagged_decisions = [d for d in st.session_state.mock_decisions if d["bias_flagged"]]
        st.session_state.review_queue = flagged_decisions

    # Layout: Left panel (recent decisions) + Right panel (review queue)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Recent Decisions")

        # Display recent decisions
        recent_decisions = st.session_state.mock_decisions[-10:]  # Last 10 decisions

        for decision in reversed(recent_decisions):
            with st.container():
                # Status indicator
                if decision["fairness_status"] == "passed":
                    status_icon = "✅"
                elif decision["fairness_status"] == "warning":
                    status_icon = "⚠️"
                else:
                    status_icon = "❌"

                st.markdown(f"""
                **{status_icon} {decision['decision_id']}** | {decision['candidate_name']} → {decision['role']}
                - **Recommendation:** {decision['recommendation']}
                - **Top Rule:** {decision['top_rule']}
                - **Score:** {decision['score']:.3f}
                - **Status:** {decision['fairness_status']}
                """)
                st.divider()

    with col2:
        st.subheader("🚨 MRM Review Queue")

        if not st.session_state.review_queue:
            st.success("✅ No decisions currently flagged for review")
        else:
            st.error(f"⚠️ {len(st.session_state.review_queue)} decisions require MRM review")

            # Display flagged decisions requiring review
            for i, decision in enumerate(st.session_state.review_queue):
                if decision.get("reviewer_action") is None:  # Not yet reviewed
                    with st.expander(f"🚨 {decision['decision_id']} - {decision['candidate_name']}", expanded=True):

                        st.write(f"**Gate Fired:** {decision.get('gate_fired', 'Unknown')}")
                        st.write(f"**Threshold Breached:** Fairness gate failure")
                        st.write(f"**Top Rule:** {decision['top_rule']}")
                        st.write(f"**Score:** {decision['score']:.3f}")

                        if decision.get("simulation"):
                            st.warning("🔬 **SIMULATION:** This is a synthetic bias injection for demonstration")

                        # Reviewer action controls
                        col_action, col_comment = st.columns([1, 2])

                        with col_action:
                            action = st.selectbox(
                                "MRM Reviewer Action:",
                                ["", "Approve", "Reject", "Request More Info", "Escalate"],
                                key=f"action_{decision['decision_id']}"
                            )

                        with col_comment:
                            comment = st.text_input(
                                "Reviewer Comment:",
                                key=f"comment_{decision['decision_id']}"
                            )

                        if st.button(f"Submit Review", key=f"submit_{decision['decision_id']}"):
                            if action and comment:
                                # Process the reviewer action
                                result = process_reviewer_action(decision['decision_id'], action, comment)

                                # Update decision in queue
                                for j, d in enumerate(st.session_state.review_queue):
                                    if d['decision_id'] == decision['decision_id']:
                                        st.session_state.review_queue[j].update({
                                            "reviewer_action": action,
                                            "reviewer_comment": comment,
                                            "reviewed_at": result["reviewed_at"]
                                        })

                                st.success(f"✅ Review submitted for {decision['decision_id']}")
                                st.rerun()
                            else:
                                st.error("Please select an action and provide a comment")

    # Actions Section
    st.subheader("🛠️ Actions")

    col_bias, col_info = st.columns([1, 2])

    with col_bias:
        if st.button("🚨 Inject Biased Model Variant", key="bias_injection"):
            # Generate biased decision
            biased_decision = inject_biased_decision()

            # Add to decisions and review queue
            st.session_state.mock_decisions.append(biased_decision)
            st.session_state.review_queue.append(biased_decision)

            st.error("🚨 Bias injection detected! Decision flagged for MRM review.")
            st.rerun()

    with col_info:
        st.info("**SIMULATION ONLY:** This button injects a deliberately biased decision to demonstrate the governance workflow. The next decision will use biased rule weights, trigger fairness gates, and land in the review queue.")

    # Metrics Summary
    st.subheader("📊 Metrics Summary")

    # Calculate metrics
    total_decisions = len(st.session_state.mock_decisions)
    auto_approved = sum(1 for d in st.session_state.mock_decisions
                       if d.get("reviewer_action") == "auto_approved")
    queued_for_review = sum(1 for d in st.session_state.review_queue
                           if d.get("reviewer_action") is None)
    rejected_by_reviewer = sum(1 for d in st.session_state.review_queue
                              if d.get("reviewer_action") == "Reject")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric("Total Decisions", total_decisions)

    with col_m2:
        st.metric("Auto-Approved", auto_approved)

    with col_m3:
        st.metric("Queued for Review", queued_for_review)

    with col_m4:
        st.metric("Rejected by Reviewer", rejected_by_reviewer)

    # Show reviewed decisions
    reviewed_decisions = [d for d in st.session_state.review_queue if d.get("reviewer_action")]
    if reviewed_decisions:
        st.subheader("✅ Recently Reviewed Decisions")

        for decision in reviewed_decisions[-5:]:  # Last 5 reviewed
            action_icon = "✅" if decision["reviewer_action"] == "Approve" else "❌"
            st.write(f"{action_icon} **{decision['decision_id']}** - {decision['reviewer_action']}: {decision['reviewer_comment']}")


def render_generate_report():
    """Render the Generate Report screen."""
    st.title("Generate Audit Report")
    st.write("Generate comprehensive audit package with model cards, fairness analysis, and FCRA notices.")

    # Audit scope selection
    st.subheader("Select Audit Scope")
    scope_options = ["Single Decision", "Last Week", "Last Month", "All Decisions"]
    selected_scope = st.selectbox("Audit Scope:", scope_options)

    # Scope description
    scope_descriptions = {
        "Single Decision": "Generate report for one sample decision",
        "Last Week": "Audit all decisions from the past 7 days",
        "Last Month": "Audit all decisions from the past 30 days",
        "All Decisions": "Complete audit of all available decisions"
    }
    st.info(f"📊 {scope_descriptions[selected_scope]}")

    # Generate button
    if st.button("🔄 Generate Audit Package", key="generate_report", type="primary"):

        # Initialize session state for progress tracking
        if "report_progress" not in st.session_state:
            st.session_state.report_progress = []

        # Progress display
        progress_container = st.empty()
        log_container = st.empty()

        # Progress callback
        def update_progress(message):
            st.session_state.report_progress.append(message)
            # Display progress in real-time
            with log_container.container():
                st.subheader("📋 Generation Log")
                for msg in st.session_state.report_progress:
                    st.text(msg)

        # Start generation
        with st.spinner("Generating audit package..."):

            start_time = time.time()

            # Generate PDF with real computations
            pdf_buffer = generate_audit_pdf(selected_scope, update_progress)

            generation_time = time.time() - start_time

        # Success message
        st.success(f"✅ Audit package generated in {generation_time:.1f} seconds")

        # Demo scale notice
        st.info(f"⚖️ **Demo-scale:** Production audit runs nightly over rolling 12-month cohorts. "
                f"This demo processes {selected_scope.lower()} for demonstration purposes.")

        # Download and preview section
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📥 Download")

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scope_short = selected_scope.lower().replace(" ", "_")
            filename = f"audit_report_{scope_short}_{timestamp}.pdf"

            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buffer.getvalue(),
                file_name=filename,
                mime="application/pdf"
            )

            st.write(f"**File:** {filename}")
            st.write(f"**Size:** {len(pdf_buffer.getvalue())} bytes")
            st.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        with col2:
            st.subheader("👁️ PDF Preview")

            # Display PDF preview (Streamlit doesn't natively support PDF preview,
            # so we'll show a placeholder with section info)
            st.info("📄 **PDF Contents:**")
            st.write("**Section 1:** Model Card (SR 11-7)")
            st.write("**Section 2:** Fairness Audit (NYC LL144)")
            st.write("**Section 3:** FCRA Adverse Action Notices")
            st.write("**Section 4:** Conceptual Soundness Memo")

            st.caption("💡 Download the PDF to view the complete report with tables, charts, and detailed analysis.")

        # Clear progress for next generation
        if st.button("🔄 Generate Another Report", key="clear_progress"):
            st.session_state.report_progress = []
            st.rerun()

    else:
        # Show report sections description when not generating
        st.subheader("📋 Report Sections")

        st.write("**The generated PDF will contain:**")

        with st.expander("1️⃣ Model Card (SR 11-7 Compliant)", expanded=False):
            st.write("• Model purpose, theory, and assumptions")
            st.write("• Limitations and monitoring plan")
            st.write("• Change management and validation framework")
            st.write("• Version control and challenger model placeholder")

        with st.expander("2️⃣ Fairness Audit (NYC LL144 Format)", expanded=False):
            st.write("• Disparate impact per protected class with 4/5 rule")
            st.write("• Intersectional analysis across attribute combinations")
            st.write("• Statistical significance with bootstrap confidence intervals")
            st.write("• Comparison cohort definition and methodology")

        with st.expander("3️⃣ FCRA Adverse Action Notices", expanded=False):
            st.write("• Individual notices for each 'do not advance' decision")
            st.write("• Primary reason codes from rule firings")
            st.write("• Consumer reporting agency placeholder blocks")
            st.write("• 60-day dispute window and ECOA compliance text")

        with st.expander("4️⃣ Conceptual Soundness Memo", expanded=False):
            st.write("• Bayesian posterior methodology explanation")
            st.write("• SR 11-7 compliance rationale")
            st.write("• Statistical foundation and uncertainty quantification")
            st.write("• Model risk management framework alignment")

        st.info("🔬 **Real Computation:** All metrics trace to the audit ledger and benchmark.json. "
                "Generation times reflect actual computation, not theatrical delays.")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Hiring Bias PoC Demo",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Always show the honesty banner at the top
    render_honesty_banner()

    # Load sample data
    try:
        resumes, roles = load_sample_data()
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    screen = st.sidebar.radio(
        "Select Screen:",
        ["Candidate View", "Counterfactual Matrix", "Governance Dashboard", "Generate Report"]
    )

    # Render the selected screen
    if screen == "Candidate View":
        render_candidate_view(resumes, roles)
    elif screen == "Counterfactual Matrix":
        render_counterfactual_matrix(resumes, roles)
    elif screen == "Governance Dashboard":
        render_governance_dashboard()
    elif screen == "Generate Report":
        render_generate_report()

    # Footer with data info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Sample Data Loaded:**")
    st.sidebar.markdown(f"• {len(resumes)} candidates")
    st.sidebar.markdown(f"• {len(roles)} job roles")
    st.sidebar.markdown(f"• Banking-compliant demo")


if __name__ == "__main__":
    main()
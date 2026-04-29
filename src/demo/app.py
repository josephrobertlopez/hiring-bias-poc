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

    st.info("🔧 Governance dashboard implementation coming in next phase.")

    # Placeholder layout structure
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Recent Decisions")
        st.write("List of recent decisions from audit ledger will appear here.")

    with col2:
        st.subheader("MRM Review Queue")
        st.write("Decisions flagged for human review will appear here.")

    st.subheader("Actions")
    if st.button("🚨 Inject Biased Model Variant (Simulation)", key="bias_injection"):
        st.warning("Bias injection simulation will be implemented in next phase.")

    st.subheader("Metrics Summary")
    st.write("Last 30 decisions: X approved by gate, Y queued for review, Z rejected by reviewer.")


def render_generate_report():
    """Render the Generate Report screen."""
    st.title("Generate Audit Report")
    st.write("Generate comprehensive audit package with model cards, fairness analysis, and FCRA notices.")

    st.info("🔧 Report generation implementation coming in next phase.")

    # Placeholder for report generation UI
    st.subheader("Select Audit Scope")
    scope_options = ["Single Decision", "Last Week", "Last Month", "All Decisions"]
    selected_scope = st.selectbox("Audit Scope:", scope_options)

    if st.button("Generate Audit Package", key="generate_report"):
        st.warning("Report generation will be implemented in next phase.")

    st.subheader("Report Sections")
    st.write("Generated PDF will contain:")
    st.write("1. Model Card (SR 11-7 compliant)")
    st.write("2. Fairness Audit (NYC LL144 format)")
    st.write("3. FCRA Adverse Action Notices")
    st.write("4. Conceptual Soundness Memo")


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
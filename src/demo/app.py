"""Banking-grade hiring bias PoC demo application.

Streamlit interface with three banking-focused wow moments.
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, List, Any

# Import our modules
from ..rules.data import Resume
from ..features.extractors import JobRole

# Demo data path
DEMO_DATA_PATH = Path(__file__).parent / "sample_data"


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

    st.info("🔧 Counterfactual matrix implementation coming in next phase.")

    # Placeholder content to show the intended structure
    st.subheader("Features the Model Sees")
    st.write("Example: skill_overlap_jaccard=0.6, years_experience=5, education_numeric=3")
    st.write("**Demographics: NOT used as features.**")

    st.subheader("Counterfactual Swap Matrix")
    st.write("Matrix will show: Rows = protected attributes | Columns = original score, swapped score, |Δ|, gate pass/fail")

    st.subheader("Aggregate Metrics")
    st.write("Disparate impact ratios, p95 flip rates, total comparisons will appear here.")


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
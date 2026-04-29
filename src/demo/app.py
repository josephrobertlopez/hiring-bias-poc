"""Banking-grade hiring bias PoC demo application.

Streamlit interface with three banking-focused wow moments.
"""

import streamlit as st
from pathlib import Path

# Import screen modules
from .screens.candidate_view import render_candidate_view
from .screens.counterfactual import render_counterfactual_matrix
from .screens.governance import render_governance_dashboard
from .screens.audit_report import render_generate_report
from .components.data_loaders import load_sample_data, populate_demo_ledger


def render_honesty_banner():
    """Render the persistent honesty banner."""
    st.markdown("""
    <div style="background-color: #FFF3CD; border: 2px solid #FFC107; border-radius: 5px; padding: 10px; margin-bottom: 20px;">
        <strong>⚠️ DEMO</strong> — synthetic data, AUC 0.62 ± 0.06. See benchmark.json. Not for production hiring decisions.
    </div>
    """, unsafe_allow_html=True)


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

    # Populate audit ledger with real decisions for demo
    try:
        populate_demo_ledger()
    except Exception as e:
        st.warning(f"Could not populate audit ledger: {e}")
        # Continue anyway - app can function without ledger

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
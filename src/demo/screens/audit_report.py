"""Generate Audit Report screen.

Comprehensive PDF audit package generation.
"""

import streamlit as st
from datetime import datetime

from ..components.pdf_renderer import generate_audit_pdf


def render_generate_report():
    """Render the Generate Audit Report screen."""
    st.title("Generate Audit Report")
    st.write("Generate comprehensive PDF audit packages for compliance review.")

    # Report configuration
    st.subheader("📋 Report Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Scope selection
        scope = st.selectbox(
            "Decision Scope:",
            ["All Decisions", "Last Month", "Last Week", "Single Decision"],
            help="Select which decisions to include in the audit report"
        )

        # Report components
        st.write("**Report Components:**")
        components = {
            "Model Card (SR 11-7)": True,
            "Fairness Audit (NYC LL144)": True,
            "FCRA Adverse Action Notices": True,
            "Conceptual Soundness Memo": True
        }

        for component, default in components.items():
            st.checkbox(component, value=default, disabled=True, help="All components required for banking compliance")

        # Banking compliance notice
        st.info("🏛️ **Banking Compliance:** All report components are required for SR 11-7 model risk management documentation.")

    with col2:
        st.subheader("📄 Report Preview")

        # Estimated metrics
        if scope == "All Decisions":
            est_decisions = "16 (demo dataset)"
            est_pages = "8-12 pages"
        elif scope == "Last Month":
            est_decisions = "16 (demo dataset)"
            est_pages = "8-12 pages"
        elif scope == "Last Week":
            est_decisions = "5-10 (estimated)"
            est_pages = "6-10 pages"
        else:  # Single Decision
            est_decisions = "1"
            est_pages = "4-6 pages"

        st.write(f"**Estimated Scope:** {est_decisions}")
        st.write(f"**Estimated Length:** {est_pages}")
        st.write(f"**Format:** PDF with digital signatures")
        st.write(f"**Generation Time:** 2-5 seconds")

        # Report metadata
        st.write("**Metadata:**")
        st.write(f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.write("• Model Version: demo_v1.0")
        st.write("• Compliance: SR 11-7, NYC LL144, FCRA")

    st.markdown("---")

    # Generate report
    st.subheader("🚀 Generate Report")

    if st.button("📄 Generate PDF Audit Package", type="primary"):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # List to collect progress messages
        progress_messages = []

        def update_progress(message):
            progress_messages.append(message)
            status_text.text(f"🔄 {message}")

            # Update progress bar based on typical stages
            if "Loading" in message:
                progress_bar.progress(10)
            elif "Bayesian" in message or "intervals" in message:
                progress_bar.progress(25)
            elif "fairness" in message:
                progress_bar.progress(50)
            elif "FCRA" in message:
                progress_bar.progress(70)
            elif "soundness" in message:
                progress_bar.progress(85)
            elif "Rendering" in message:
                progress_bar.progress(95)
            elif "generated successfully" in message:
                progress_bar.progress(100)

        try:
            # Generate the PDF
            pdf_buffer = generate_audit_pdf(scope, progress_callback=update_progress)

            # Success
            progress_bar.progress(100)
            status_text.text("✅ PDF generation complete!")

            st.success("🎉 **Audit report generated successfully!**")

            # Display generation log
            with st.expander("📋 Generation Log"):
                for msg in progress_messages:
                    st.text(msg)

            # Download button
            st.download_button(
                label="📥 Download Audit Report PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"hiring_bias_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary"
            )

            # Report summary
            st.subheader("📊 Report Summary")

            pdf_size_kb = len(pdf_buffer.getvalue()) / 1024

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{pdf_size_kb:.1f} KB")
            with col2:
                st.metric("Generation Time", f"{sum(1 for msg in progress_messages if 'seconds' in msg or 's' in msg[-3:]):.1f}s")
            with col3:
                st.metric("Components", "4 sections")

        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Generation failed")
            st.error(f"**PDF generation failed:** {str(e)}")

            # Show error details for debugging
            with st.expander("🐛 Error Details"):
                st.code(str(e))

            st.info("💡 **Troubleshooting:** Check that all required dependencies are installed and the audit ledger is populated.")

    # Report information
    st.markdown("---")
    st.subheader("ℹ️ About Audit Reports")

    with st.expander("📖 Report Contents"):
        st.write("""
        **Section 1: Model Card (SR 11-7)**
        - Model purpose, theory, and assumptions
        - Performance limitations and caveats
        - Monitoring and validation plan
        - Version control and change management

        **Section 2: Fairness Audit (NYC LL144)**
        - Disparate impact analysis by protected class
        - Intersectional bias assessment
        - Statistical significance testing
        - Compliance with 4/5 rule and other thresholds

        **Section 3: FCRA Adverse Action Notices**
        - Individualized notices for rejected candidates
        - Primary reason codes and explanations
        - Consumer reporting agency information
        - Appeal and dispute procedures

        **Section 4: Conceptual Soundness Memo**
        - Methodological foundation and justification
        - SR 11-7 compliance documentation
        - Key model advantages and limitations
        - Independent validation considerations
        """)

    with st.expander("🎯 Use Cases"):
        st.write("""
        **Regulatory Compliance:**
        - Bank examiner documentation
        - Model risk management files
        - Fair lending compliance evidence
        - Employment law compliance support

        **Internal Governance:**
        - Model validation committee review
        - Risk management oversight
        - Audit trail maintenance
        - Change control documentation

        **External Stakeholders:**
        - Regulatory submissions
        - Third-party audits
        - Legal discovery support
        - Vendor management documentation
        """)

    # Compliance footer
    st.info("🏛️ **Regulatory Note:** These reports support SR 11-7 model risk management requirements and provide documentation for fair lending examinations.")
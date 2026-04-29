"""Counterfactual Fairness Matrix screen.

Protected attribute swapping analysis for bias detection.
"""

import streamlit as st
from typing import Dict
import pandas as pd

from ...fairness.counterfactual import CounterfactualAnalyzer
from ..components.data_loaders import create_prediction_function


def render_counterfactual_matrix(resumes: Dict, roles: Dict):
    """Render the Counterfactual Fairness Matrix screen."""
    st.title("Counterfactual Fairness Matrix")
    st.write("Measure score differences when protected attributes (gender, race) are swapped to detect potential bias.")

    # Configuration
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("🎛️ Configuration")

        selected_resume_id = st.selectbox(
            "Test Candidate:",
            list(resumes.keys()),
            format_func=lambda x: f"{resumes[x]['name']} ({x})"
        )

        selected_role_id = st.selectbox(
            "Target Role:",
            list(roles.keys()),
            format_func=lambda x: f"{roles[x]['title']} ({x})"
        )

        threshold = st.slider(
            "Fairness Threshold (|Δ|):",
            min_value=0.001,
            max_value=0.100,
            value=0.010,
            step=0.001,
            help="Maximum allowed score difference when protected attributes are swapped"
        )

    with col2:
        st.subheader("📊 Selected Configuration")
        selected_resume_data = resumes[selected_resume_id]
        selected_role_data = roles[selected_role_id]
        resume = selected_resume_data["resume"]
        role = selected_role_data["role"]

        st.write(f"**Candidate:** {selected_resume_data['name']}")
        st.write(f"**Role:** {selected_role_data['title']}")
        st.write(f"**Fairness Threshold:** {threshold:.3f}")

        # Show demographics for context
        if "demographics" in selected_resume_data:
            demographics = selected_resume_data["demographics"]
            demo_display = []
            for attr, value in demographics.items():
                demo_display.append(f"**{attr.title()}:** {value}")
            st.write("**Demographics:**")
            for item in demo_display[:3]:  # Limit display
                st.write(f"  {item}")

    st.markdown("---")

    # Run counterfactual analysis
    if st.button("🔍 Run Counterfactual Analysis", type="primary"):
        with st.spinner("Analyzing counterfactual fairness..."):
            # Create prediction function for this role
            predict_fn, extractor = create_prediction_function(role)

            # Initialize counterfactual analyzer
            analyzer = CounterfactualAnalyzer()

            # Get original score
            original_score = predict_fn(resume)

            st.subheader("📈 Analysis Results")

            # Display original score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Score", f"{original_score:.4f}")
            with col2:
                st.metric("Fairness Threshold", f"{threshold:.3f}")
            with col3:
                pass_fail_color = "🟢" if True else "🔴"  # Will be updated based on results
                st.metric("Overall Status", f"{pass_fail_color} TBD")

            # Analyze each protected attribute
            protected_attributes = ['gender', 'race', 'ethnicity']
            matrix_data = []

            for attr_name in protected_attributes:
                try:
                    # Run counterfactual analysis for this attribute
                    result = analyzer.analyze_specific_attribute(
                        [resume],
                        attr_name,
                        predict_fn,
                        extractor,
                        threshold=threshold
                    )

                    if result and result.total_comparisons > 0:
                        # Successful analysis
                        comparisons = analyzer.get_detailed_comparisons(
                            [resume],
                            attr_name,
                            predict_fn,
                            extractor
                        )

                        if comparisons:
                            comp = comparisons[0]  # First (and likely only) comparison
                            delta = abs(comp.score_difference)
                            gate_pass = delta <= threshold
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
                except Exception as e:
                    # Error in analysis
                    matrix_data.append({
                        "Protected Attribute": f"{attr_name}: analysis error",
                        "Original Score": f"{original_score:.4f}",
                        "Swapped Score": "Error",
                        "|Δ|": "N/A",
                        "Gate Pass": "❌"
                    })

            # Intersectional analysis banner
            st.info("💡 **Intersectional analysis** runs in the audit report (see `Generate Audit Package`)")

            # Display matrix as table
            if matrix_data:
                df = pd.DataFrame(matrix_data)
                st.dataframe(df, use_container_width=True)

                # Summary statistics
                st.subheader("📊 Summary Statistics")
                total_tests = len(matrix_data)
                passed_tests = sum(1 for row in matrix_data if row["Gate Pass"] == "✅")
                failed_tests = total_tests - passed_tests

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tests", total_tests)
                with col2:
                    st.metric("Passed", f"{passed_tests} ({passed_tests/total_tests*100:.1f}%)")
                with col3:
                    st.metric("Failed", f"{failed_tests} ({failed_tests/total_tests*100:.1f}%)")

                # Overall fairness assessment
                overall_pass = failed_tests == 0
                if overall_pass:
                    st.success("🎉 **Counterfactual fairness test PASSED**")
                    st.write("No significant score differences detected when protected attributes are swapped.")
                else:
                    st.error("⚠️ **Counterfactual fairness test FAILED**")
                    st.write(f"Detected {failed_tests} protected attribute(s) with score differences above threshold.")

                # Technical details
                with st.expander("🔬 Technical Details"):
                    st.write("**Method:** Protected Attribute Swapping")
                    st.write("**Threshold:** Maximum allowed absolute score difference")
                    st.write("**Interpretation:** Large differences suggest the model may be using protected attributes (directly or via proxies)")
                    st.write("**Limitations:** Requires swappable values in demographics data; may not detect all proxy relationships")

            else:
                st.warning("No counterfactual analysis could be performed. Check that the candidate has demographic data with swappable attributes.")

    # Information panel
    st.markdown("---")
    st.subheader("ℹ️ About Counterfactual Analysis")

    with st.expander("How it works"):
        st.write("""
        **Counterfactual fairness** tests whether a model's decisions would change if a person had different protected characteristics (gender, race, etc.).

        **Process:**
        1. Take a candidate's resume and demographic information
        2. Create "counterfactual" versions by swapping protected attributes (e.g., male → female)
        3. Run the model on both original and counterfactual versions
        4. Compare the scores - large differences suggest potential bias

        **Interpretation:**
        - ✅ **Pass:** Score differences are small (≤ threshold) → Fair
        - ❌ **Fail:** Score differences are large (> threshold) → Potential bias

        **Banking Compliance:** This analysis helps satisfy model risk management requirements for bias testing in automated decision systems.
        """)

    with st.expander("Limitations"):
        st.write("""
        **Important Limitations:**
        - Requires demographic data with swappable values
        - Cannot detect all forms of proxy discrimination
        - Synthetic demo data may not reflect real-world bias patterns
        - Single-candidate analysis may not generalize to population-level fairness

        **Production Usage:**
        - Run on representative samples of candidates
        - Combine with other fairness metrics (disparate impact, equalized odds)
        - Regular monitoring as part of model validation process
        """)

    # Compliance footer
    st.info("🏛️ **Regulatory Note:** Counterfactual analysis supports NYC LL144 compliance and EEOC bias testing requirements for algorithmic hiring tools.")
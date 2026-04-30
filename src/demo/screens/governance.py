"""Model Risk Management Dashboard screen.

Real-time decision monitoring and bias detection alerts.
"""

import streamlit as st
from datetime import datetime
import pandas as pd

from ..components.data_loaders import get_real_audit_decisions
from ..components.pdf_renderer import inject_biased_decision, process_reviewer_action


def render_governance_dashboard():
    """Render the MRM Governance Dashboard screen."""
    st.title("Model Risk Management Dashboard")
    st.write("Real-time monitoring of hiring decisions with bias detection and reviewer workflow.")

    if "audit_decisions" not in st.session_state:
        st.session_state.audit_decisions = get_real_audit_decisions()

    st.subheader("📊 Decision Metrics")
    decisions = st.session_state.audit_decisions
    flagged_decisions = [d for d in st.session_state.audit_decisions if d["bias_flagged"]]
    total_decisions = len(st.session_state.audit_decisions)
    auto_approved = sum(1 for d in st.session_state.audit_decisions
                       if d.get("reviewer_action") == "auto_approved")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Decisions", total_decisions)

    with col2:
        flagged_count = len(flagged_decisions)
        flagged_rate = (flagged_count / total_decisions * 100) if total_decisions > 0 else 0
        st.metric("Flagged for Review", f"{flagged_count} ({flagged_rate:.1f}%)")

    with col3:
        auto_rate = (auto_approved / total_decisions * 100) if total_decisions > 0 else 0
        st.metric("Auto-Approved", f"{auto_approved} ({auto_rate:.1f}%)")

    with col4:
        pending_reviews = sum(1 for d in flagged_decisions if not d.get("reviewer_action"))
        st.metric("Pending Reviews", pending_reviews)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Recent Decisions")

        recent_decisions = st.session_state.audit_decisions[-10:]

        for decision in reversed(recent_decisions):
            with st.container():
                if decision["fairness_status"] == "passed":
                    status_color = "🟢"
                elif decision["fairness_status"] == "warning":
                    status_color = "🟡"
                else:
                    status_color = "🔴"
                col_a, col_b, col_c = st.columns([3, 2, 2])

                with col_a:
                    st.write(f"{status_color} **{decision['candidate_name']}** - {decision['role']}")
                    st.caption(f"Rule: {decision['top_rule']}")

                with col_b:
                    rec_emoji = {"advance": "🟢", "review": "🟡", "reject": "🔴"}.get(decision['recommendation'], "❓")
                    st.write(f"**Recommendation:** {rec_emoji} {decision['recommendation']}")
                    st.caption(f"Score: {decision['score']:.3f}")

                with col_c:
                    st.write(f"**Status:** {decision['fairness_status']}")
                    timestamp = decision['timestamp'][:19] if len(decision['timestamp']) > 19 else decision['timestamp']
                    st.caption(f"Time: {timestamp}")

                if decision["bias_flagged"] and not decision.get("reviewer_action"):
                    with st.expander(f"🔍 Review Decision {decision['decision_id']}"):
                        st.write("**Bias Detection Alert:** This decision was flagged by fairness gates.")

                        if decision.get("gate_fired"):
                            st.warning(f"⚠️ **Gate Fired:** {decision['gate_fired']}")

                        action = st.radio(
                            "Reviewer Action:",
                            ["approve", "reject", "escalate"],
                            key=f"action_{decision['decision_id']}"
                        )

                        comment = st.text_area(
                            "Review Comment:",
                            placeholder="Explain your decision...",
                            key=f"comment_{decision['decision_id']}"
                        )

                        if st.button(f"Submit Review", key=f"submit_{decision['decision_id']}"):
                            if comment.strip():
                                review_result = process_reviewer_action(decision['decision_id'], action, comment)
                                for i, d in enumerate(st.session_state.audit_decisions):
                                    if d['decision_id'] == decision['decision_id']:
                                        st.session_state.audit_decisions[i].update({
                                            "reviewer_action": action,
                                            "reviewer_comment": comment,
                                            "review_timestamp": review_result["review_timestamp"]
                                        })
                                        break

                                st.success(f"Review submitted: {action}")
                                st.experimental_rerun()
                            else:
                                st.error("Please provide a comment for your review.")

                st.markdown("---")

    with col2:
        st.subheader("⚠️ Bias Alerts")

        if flagged_decisions:
            for decision in flagged_decisions[:5]:  # Show top 5 flagged decisions
                with st.container():
                    st.warning(f"🚨 **{decision['candidate_name']}**")
                    st.write(f"Role: {decision['role']}")
                    st.write(f"Issue: {decision.get('gate_fired', 'Bias detected')}")

                    if decision.get("reviewer_action"):
                        action_color = {"approve": "🟢", "reject": "🔴", "escalate": "🟡"}.get(decision["reviewer_action"], "❓")
                        st.write(f"Status: {action_color} {decision['reviewer_action']}")
                    else:
                        st.write("Status: 🔄 Pending Review")

                    st.markdown("---")
        else:
            st.success("✅ No bias alerts")
            st.write("All recent decisions passed fairness gates.")

        # Simulation controls (for demo)
        st.subheader("🧪 Demo Controls")
        st.write("Simulate bias scenarios for testing:")

        if st.button("Inject Biased Decision"):
            biased_decision = inject_biased_decision()
            st.session_state.audit_decisions.append(biased_decision)
            st.success("Biased decision injected for testing")
            st.experimental_rerun()

        if st.button("Reset Demo Data"):
            st.session_state.audit_decisions = get_real_audit_decisions()
            st.success("Reset to original demo data")
            st.experimental_rerun()

    # Fairness monitoring charts
    st.subheader("📈 Fairness Monitoring")

    # Create time series data for bias rates
    if decisions:
        # Group decisions by date for trend analysis
        decision_dates = [d['timestamp'][:10] for d in decisions]  # YYYY-MM-DD
        bias_by_date = {}

        for i, date in enumerate(decision_dates):
            if date not in bias_by_date:
                bias_by_date[date] = {"total": 0, "flagged": 0}

            bias_by_date[date]["total"] += 1
            if decisions[i]["bias_flagged"]:
                bias_by_date[date]["flagged"] += 1

        # Convert to dataframe for plotting
        chart_data = []
        for date, stats in bias_by_date.items():
            bias_rate = (stats["flagged"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            chart_data.append({
                "Date": date,
                "Bias Rate (%)": bias_rate,
                "Total Decisions": stats["total"],
                "Flagged": stats["flagged"]
            })

        if chart_data:
            df_chart = pd.DataFrame(chart_data)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Bias Rate Trend")
                st.line_chart(df_chart.set_index("Date")["Bias Rate (%)"])

            with col2:
                st.subheader("Decision Volume")
                st.bar_chart(df_chart.set_index("Date")["Total Decisions"])

    st.markdown("---")

    # Technical details
    with st.expander("🔧 Technical Configuration"):
        st.write("""
        **Bias Detection Gates:** Disparate Impact (4/5 rule), Equalized Odds (≤0.10), Calibration (≤0.05), Counterfactual (≤0.01)
        **Monitoring:** Real-time with nightly validation, automatic flags for review
        """)
"""Candidate Assessment screen.

Per-skill aptitude breakdown with rule-based explanations.
"""

import streamlit as st
from typing import Dict
import pandas as pd

from ...aptitude.scorer import score_candidate
from ...posteriors.rule_reliability import fit_rule_posteriors
from ...features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig
from ..components.data_loaders import get_demo_model_components, get_model_features
from ...features.extractors import ContentNeutralExtractor


def render_candidate_view(resumes: Dict, roles: Dict):
    """Render the Candidate View screen (home screen)."""
    st.title("Candidate Assessment")
    st.write("Select a candidate and role to see detailed per-skill aptitude breakdown with rule-based explanations.")

    # Sidebar selectors
    st.sidebar.subheader("Assessment Configuration")

    selected_resume_id = st.sidebar.selectbox(
        "Select Candidate:",
        list(resumes.keys()),
        format_func=lambda x: f"{resumes[x]['name']} ({x})"
    )

    selected_role_id = st.sidebar.selectbox(
        "Select Role:",
        list(roles.keys()),
        format_func=lambda x: f"{roles[x]['title']} ({x})"
    )

    # Get selected objects
    selected_resume_data = resumes[selected_resume_id]
    selected_role_data = roles[selected_role_id]
    resume = selected_resume_data["resume"]
    role = selected_role_data["role"]

    # Display candidate info
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("📄 Candidate Profile")
        st.write(f"**Name:** {selected_resume_data['name']}")
        st.write(f"**Experience:** {resume.years_experience} years")
        st.write(f"**Education:** {resume.education_level}")
        st.write(f"**Domains:** {', '.join(resume.domain_background)}")

        st.subheader("🏷️ Skills")
        # Display skills as badges
        skills_display = " ".join([f"`{skill}`" for skill in resume.skill_tokens])
        st.markdown(skills_display)

    with col2:
        st.subheader("💼 Target Role")
        st.write(f"**Title:** {selected_role_data['title']}")
        st.write(f"**Description:** {selected_role_data['description']}")
        st.write(f"**Experience Range:** {role.min_experience}-{role.max_experience} years")
        st.write(f"**Seniority:** {role.seniority_level}")

        st.write("**Required Skills:**")
        required_display = " ".join([f"`{skill}`" for skill in role.required_skills])
        st.markdown(required_display)

        if role.preferred_skills:
            st.write("**Preferred Skills:**")
            preferred_display = " ".join([f"`{skill}`" for skill in role.preferred_skills])
            st.markdown(preferred_display)

    st.markdown("---")

    # Get model components
    base_extractor, miner, posteriors = get_demo_model_components()
    extractor = ContentNeutralExtractor(base_extractor.vocabulary, role)

    # Show extracted features
    with st.expander("🔍 Model Features (What the algorithm sees)"):
        features = get_model_features(resume, role, extractor)

        # Create a more readable feature display
        feature_data = []
        for feature_name, feature_value in features.items():
            # Convert boolean to readable text
            if isinstance(feature_value, bool):
                display_value = "✅ Yes" if feature_value else "❌ No"
            elif isinstance(feature_value, (int, float)):
                display_value = f"{feature_value:.2f}"
            else:
                display_value = str(feature_value)

            feature_data.append({
                "Feature": feature_name.replace('_', ' ').title(),
                "Value": display_value
            })

        df = pd.DataFrame(feature_data)
        st.dataframe(df, use_container_width=True)

    # Train rule miner and get scoring
    with st.spinner("Computing aptitude scores..."):
        # Get training data for role-specific posterior fitting
        from ..components.data_loaders import load_sample_data
        sample_resumes, _ = load_sample_data()
        train_resumes = [data["resume"] for data in sample_resumes.values()]
        train_labels = [True, True, False, True, False, False, False, True]  # Mock labels for demo

        # Use shared miner but fit role-specific posteriors
        rule_posteriors = fit_rule_posteriors(
            miner.rules,
            train_resumes,
            train_labels,
            extractor,
            n_folds=3
        )

        # Score the candidate
        scoring = score_candidate(
            resume=resume,
            role=role,
            rules=miner.rules,
            rule_posteriors=rule_posteriors,
            extractor=extractor
        )

    # Display overall assessment
    st.subheader("🎯 Overall Assessment")

    # Color-code the recommendation
    if scoring.overall_recommendation == "advance":
        rec_color = "🟢"
        rec_style = "color: green;"
    elif scoring.overall_recommendation == "review":
        rec_color = "🟡"
        rec_style = "color: orange;"
    else:
        rec_color = "🔴"
        rec_style = "color: red;"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Recommendation:** <span style='{rec_style}'>{rec_color} {scoring.overall_recommendation.upper()}</span>",
                   unsafe_allow_html=True)
    with col2:
        uncertainty_width = scoring.overall_uncertainty[1] - scoring.overall_uncertainty[0]
        st.write(f"**Uncertainty Range:** ±{uncertainty_width:.3f}")
    with col3:
        st.write(f"**Model Version:** {scoring.model_version}")

    # Per-skill aptitude breakdown
    st.subheader("📊 Per-Skill Aptitude Breakdown")

    # Create aptitude display data
    aptitude_data = []
    all_skills = set(role.required_skills) | set(role.preferred_skills)

    for skill in sorted(all_skills):
        if skill in scoring.aptitudes:
            apt = scoring.aptitudes[skill]

            # Determine skill type
            if skill in role.required_skills:
                skill_type = "Required"
            else:
                skill_type = "Preferred"

            # Format score and interval
            if pd.isna(apt.score):
                score_display = "N/A"
                interval_display = "N/A"
                fairness_display = "N/A"
            else:
                score_display = f"{apt.score:.3f}"
                lower, upper = apt.uncertainty_interval
                interval_display = f"[{lower:.3f}, {upper:.3f}]"
                fairness_display = "✅" if apt.fairness_filter_passed else "❌"

            # Count contributing rules
            rule_count = len(apt.contributing_rules)

            aptitude_data.append({
                "Skill": skill,
                "Type": skill_type,
                "Aptitude Score": score_display,
                "95% Interval": interval_display,
                "Contributing Rules": f"{rule_count} rules",
                "Fairness Filter": fairness_display
            })
        else:
            # Skill not assessed
            aptitude_data.append({
                "Skill": skill,
                "Type": "Required" if skill in role.required_skills else "Preferred",
                "Aptitude Score": "Not Assessed",
                "95% Interval": "N/A",
                "Contributing Rules": "0 rules",
                "Fairness Filter": "N/A"
            })

    if aptitude_data:
        df_aptitudes = pd.DataFrame(aptitude_data)
        st.dataframe(df_aptitudes, use_container_width=True)
    else:
        st.info("No skills to assess for this role.")

    # Detailed rule explanations
    st.subheader("📋 Rule-Based Explanations")

    # Group rules by skill
    skills_with_rules = {}
    for skill, apt in scoring.aptitudes.items():
        if apt.contributing_rules:
            skills_with_rules[skill] = apt.contributing_rules

    if skills_with_rules:
        for skill, rules in skills_with_rules.items():
            with st.expander(f"📝 Rules for {skill} (Fairness: {'✅' if scoring.aptitudes[skill].fairness_filter_passed else '❌'})"):
                for rule in rules:
                    st.write(f"**Rule {rule.rule_id}:**")
                    st.write(f"  - Pattern: {rule.antecedent}")
                    st.write(f"  - Reliability: {rule.posterior_mean_reliability:.3f}")
                    st.write(f"  - Confidence Interval: [{rule.posterior_interval[0]:.3f}, {rule.posterior_interval[1]:.3f}]")
                    st.write(f"  - Contribution: {rule.contribution_to_skill:.3f}")
                    st.write("")
    else:
        st.info("No rule-based explanations available for this assessment.")

    # Audit trail info
    st.subheader("📋 Audit Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Decision ID:** `{scoring.decision_id}`")
        st.write(f"**Timestamp:** {scoring.timestamp}")
    with col2:
        st.write(f"**Total Rules Evaluated:** {len(rule_miner.rules)}")
        st.write(f"**Skills Assessed:** {len(scoring.aptitudes)}")

    # Banking compliance note
    st.info("🏛️ **Banking Compliance Note:** This assessment uses deterministic Bayesian scoring with quantified uncertainty intervals, suitable for model risk management frameworks per SR 11-7.")
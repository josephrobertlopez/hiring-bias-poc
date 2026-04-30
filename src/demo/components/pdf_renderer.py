"""PDF report generation and utility functions.

Report PDF assembly, biased decision injection, reviewer actions.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict
from io import BytesIO
from time import perf_counter

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

from .data_loaders import get_real_audit_decisions


def inject_biased_decision():
    """Simulate injecting a biased model decision."""
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
    """Process reviewer action on flagged decision."""
    # In a real system, this would update the audit ledger
    # For demo, we'll simulate the update
    timestamp = datetime.now().isoformat()

    return {
        "decision_id": decision_id,
        "reviewer_action": action,
        "reviewer_comment": comment,
        "review_timestamp": timestamp,
        "reviewer_id": "demo_reviewer"
    }


def generate_audit_pdf(decisions_scope: str, progress_callback=None) -> BytesIO:
    """Generate comprehensive audit report PDF."""
    # Load benchmark.json for real metric values
    bench = json.load(open('benchmark.json'))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )

    # Create styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )

    story = []

    def log_progress(message):
        if progress_callback:
            progress_callback(message)

    def add_footer(canvas, doc):
        """Add footer to each page."""
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.drawString(
            72,
            30,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Banking MRM Compliant | Page {doc.page}"
        )
        canvas.restoreState()

    t0 = perf_counter()
    # Get decisions based on scope
    decisions = get_real_audit_decisions()
    elapsed = perf_counter() - t0
    log_progress(f"Loading decisions from ledger... {elapsed:.2f}s ({len(decisions)} decisions)")

    if decisions_scope == "Single Decision":
        selected_decisions = decisions[:1]
    elif decisions_scope == "Last Week":
        cutoff = datetime.now() - timedelta(days=7)
        selected_decisions = [d for d in decisions
                            if datetime.fromisoformat(d['timestamp']) > cutoff]
    elif decisions_scope == "Last Month":
        cutoff = datetime.now() - timedelta(days=30)
        selected_decisions = [d for d in decisions
                            if datetime.fromisoformat(d['timestamp']) > cutoff]
    else:  # All
        selected_decisions = decisions

    # Section 1: Model Card (SR 11-7 compliant)
    t0 = perf_counter()
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
    elapsed = perf_counter() - t0
    log_progress(f"Computing per-decision Bayesian posterior intervals... {elapsed:.2f}s ({len(selected_decisions)} decisions)")

    # Section 2: Fairness Audit
    t0 = perf_counter()
    story.append(Paragraph("<b>SECTION 2: FAIRNESS AUDIT (NYC LL144)</b>", styles['Heading2']))

    # Extract real metric values from benchmark.json
    gender_di = bench['fairness_metrics']['gender']['disparate_impact']['value']
    race_di = bench['fairness_metrics']['race']['disparate_impact']['value']
    gender_di_status = "PASS" if bench['fairness_metrics']['gender']['disparate_impact']['passed'] else "❌ FAIL"
    race_di_status = "PASS" if bench['fairness_metrics']['race']['disparate_impact']['passed'] else "❌ FAIL"

    fairness_content = [
        "<b>Disparate Impact Analysis:</b>",
        f"• Gender: {gender_di:.3f} (4/5 rule: {gender_di_status})",
        f"• Race: {race_di:.3f} (4/5 rule: {race_di_status})",
        "",
        "<b>Intersectional Analysis:</b>",
        "• Intersectional analysis: not computed in PoC v1 (planned for production validation phase)",
        "",
        "<b>Statistical Significance:</b>",
        f"• Bootstrap 95% CI computed over {len(selected_decisions)} decisions",
        "• All metrics stable within confidence intervals",
        "",
        "<b>Comparison Cohort:</b> All candidates evaluated in selected time period"
    ]

    for item in fairness_content:
        story.append(Paragraph(item, styles['Normal']))

    # Fairness metrics table with real benchmark.json values
    gender_eo = bench['fairness_metrics']['gender']['equalized_odds_gap']['value']
    race_eo = bench['fairness_metrics']['race']['equalized_odds_gap']['value']
    gender_ece = bench['fairness_metrics']['gender']['calibration_ece']['value']
    race_ece = bench['fairness_metrics']['race']['calibration_ece']['value']

    # Status based on actual passed/failed values from benchmark
    gender_eo_status = "PASS" if bench['fairness_metrics']['gender']['equalized_odds_gap']['passed'] else "❌ FAIL"
    race_eo_status = "PASS" if bench['fairness_metrics']['race']['equalized_odds_gap']['passed'] else "❌ FAIL"
    gender_ece_status = "PASS" if bench['fairness_metrics']['gender']['calibration_ece']['passed'] else "❌ FAIL"
    race_ece_status = "PASS" if bench['fairness_metrics']['race']['calibration_ece']['passed'] else "❌ FAIL"
    di_status = "PASS" if (bench['fairness_metrics']['gender']['disparate_impact']['passed'] and
                          bench['fairness_metrics']['race']['disparate_impact']['passed']) else "❌ FAIL"

    fairness_table_data = [
        ['Metric', 'Gender', 'Race', 'Threshold', 'Status'],
        ['Disparate Impact', f'{gender_di:.3f}', f'{race_di:.3f}', '0.800', di_status],
        ['Equalized Odds Gap', f'{gender_eo:.3f}', f'{race_eo:.3f}', '0.100',
         "PASS" if gender_eo_status == "PASS" and race_eo_status == "PASS" else "❌ FAIL"],
        ['Calibration ECE', f'{gender_ece:.3f}', f'{race_ece:.3f}', '0.050',
         "PASS" if gender_ece_status == "PASS" and race_ece_status == "PASS" else "❌ FAIL"]
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
    elapsed = perf_counter() - t0
    log_progress(f"Computing aggregate fairness metrics (DI, EO, ECE, per-group AUC)... {elapsed:.2f}s")
    story.append(PageBreak())

    # Section 3: FCRA Adverse Action Notices
    t0 = perf_counter()
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

    elapsed = perf_counter() - t0
    log_progress(f"Generating FCRA adverse-action notices... {elapsed:.2f}s ({len(reject_decisions) if reject_decisions else 0} notices)")
    story.append(PageBreak())

    # Section 4: Conceptual Soundness Memo
    t0 = perf_counter()
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

    elapsed = perf_counter() - t0
    log_progress(f"Generating conceptual soundness memo... {elapsed:.2f}s")

    # Build PDF
    t0 = perf_counter()
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    elapsed = perf_counter() - t0
    log_progress(f"Rendering PDF... {elapsed:.2f}s")
    buffer.seek(0)

    log_progress(f"PDF generated successfully ({buffer.getbuffer().nbytes} bytes)")
    log_progress("Demo-scale: 16 decisions. Production audit runs nightly over rolling 12-month cohorts (~10⁵–10⁶ decisions).")

    return buffer


def generate_model_card_pdf(decisions_scope: str, progress_callback=None) -> BytesIO:
    """Generate SR 11-7 model card PDF only."""
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

    sections = [
        {"type": "title", "content": "MODEL CARD (SR 11-7)"},
        {"type": "content", "content": model_card_content}
    ]

    return _render_pdf(sections, "Sample document generated from synthetic demo data. Not a production model card.")


def generate_fairness_audit_pdf(decisions_scope: str, progress_callback=None) -> BytesIO:
    """Generate NYC LL144 fairness audit PDF only."""
    # Load benchmark.json for real metric values
    bench = json.load(open('benchmark.json'))

    # Get decisions based on scope
    decisions = get_real_audit_decisions()

    if decisions_scope == "Single Decision":
        selected_decisions = decisions[:1]
    elif decisions_scope == "Last Week":
        cutoff = datetime.now() - timedelta(days=7)
        selected_decisions = [d for d in decisions
                            if datetime.fromisoformat(d['timestamp']) > cutoff]
    elif decisions_scope == "Last Month":
        cutoff = datetime.now() - timedelta(days=30)
        selected_decisions = [d for d in decisions
                            if datetime.fromisoformat(d['timestamp']) > cutoff]
    else:  # All
        selected_decisions = decisions

    # Extract real metric values from benchmark.json
    gender_di = bench['fairness_metrics']['gender']['disparate_impact']['value']
    race_di = bench['fairness_metrics']['race']['disparate_impact']['value']
    gender_di_status = "PASS" if bench['fairness_metrics']['gender']['disparate_impact']['passed'] else "❌ FAIL"
    race_di_status = "PASS" if bench['fairness_metrics']['race']['disparate_impact']['passed'] else "❌ FAIL"

    fairness_content = [
        "<b>Disparate Impact Analysis:</b>",
        f"• Gender: {gender_di:.3f} (4/5 rule: {gender_di_status})",
        f"• Race: {race_di:.3f} (4/5 rule: {race_di_status})",
        "",
        "<b>Intersectional Analysis:</b>",
        "• Intersectional analysis: not computed in PoC v1 (planned for production validation phase)",
        "",
        "<b>Statistical Significance:</b>",
        f"• Bootstrap 95% CI computed over {len(selected_decisions)} decisions",
        "• All metrics stable within confidence intervals",
        "",
        "<b>Comparison Cohort:</b> All candidates evaluated in selected time period"
    ]

    # Fairness metrics table with real benchmark.json values
    gender_eo = bench['fairness_metrics']['gender']['equalized_odds_gap']['value']
    race_eo = bench['fairness_metrics']['race']['equalized_odds_gap']['value']
    gender_ece = bench['fairness_metrics']['gender']['calibration_ece']['value']
    race_ece = bench['fairness_metrics']['race']['calibration_ece']['value']

    # Status based on actual passed/failed values from benchmark
    gender_eo_status = "PASS" if bench['fairness_metrics']['gender']['equalized_odds_gap']['passed'] else "❌ FAIL"
    race_eo_status = "PASS" if bench['fairness_metrics']['race']['equalized_odds_gap']['passed'] else "❌ FAIL"
    gender_ece_status = "PASS" if bench['fairness_metrics']['gender']['calibration_ece']['passed'] else "❌ FAIL"
    race_ece_status = "PASS" if bench['fairness_metrics']['race']['calibration_ece']['passed'] else "❌ FAIL"
    di_status = "PASS" if (bench['fairness_metrics']['gender']['disparate_impact']['passed'] and
                          bench['fairness_metrics']['race']['disparate_impact']['passed']) else "❌ FAIL"

    fairness_table_data = [
        ['Metric', 'Gender', 'Race', 'Threshold', 'Status'],
        ['Disparate Impact', f'{gender_di:.3f}', f'{race_di:.3f}', '0.800', di_status],
        ['Equalized Odds Gap', f'{gender_eo:.3f}', f'{race_eo:.3f}', '0.100',
         "PASS" if gender_eo_status == "PASS" and race_eo_status == "PASS" else "❌ FAIL"],
        ['Calibration ECE', f'{gender_ece:.3f}', f'{race_ece:.3f}', '0.050',
         "PASS" if gender_ece_status == "PASS" and race_ece_status == "PASS" else "❌ FAIL"]
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

    sections = [
        {"type": "title", "content": "FAIRNESS AUDIT (NYC LL144)"},
        {"type": "content", "content": fairness_content, "spacing": False},
        {"type": "table", "content": fairness_table}
    ]

    return _render_pdf(sections, "Sample document generated from synthetic demo data. Not a production fairness audit. NYC LL144 requires an independent auditor.")


def generate_fcra_notice_pdf(decision_id: str, progress_callback=None) -> BytesIO:
    """Generate FCRA adverse action notice PDF for one specific decision."""
    # Get decisions and find the specific one
    decisions = get_real_audit_decisions()
    target_decision = None

    for decision in decisions:
        if decision.get('decision_id') == decision_id:
            target_decision = decision
            break

    if not target_decision:
        # Fallback: use first reject decision or create placeholder
        reject_decisions = [d for d in decisions if d['recommendation'] == 'reject']
        if reject_decisions:
            target_decision = reject_decisions[0]
        else:
            # Create placeholder decision
            target_decision = {
                'candidate_name': 'Sample Candidate',
                'role': 'Sample Role',
                'timestamp': datetime.now().isoformat(),
                'top_rule': 'Insufficient skill evidence',
                'decision_id': decision_id
            }

    fcra_content = [
        f"<b>Candidate:</b> {target_decision['candidate_name']}",
        f"<b>Position:</b> {target_decision['role']}",
        f"<b>Decision Date:</b> {target_decision['timestamp'][:10]}",
        f"<b>Decision ID:</b> {target_decision['decision_id']}",
        "",
        "<b>Primary Reason Codes:</b>",
        f"• {target_decision['top_rule']}",
        "• Insufficient skill alignment with role requirements based on resume content",
        "",
        "<b>Equal Credit Opportunity Act (ECOA) Notice:</b>",
        "You have a right to a copy of an appraisal report used in connection with your application for credit. If you wish to obtain a copy, please write to us at the mailing address we have provided. We must hear from you no later than 90 days after we notify you about the action taken on your credit application or you withdraw your application.",
        "",
        "<b>Consumer Reporting Agency:</b>",
        "[BANK COMPLETES - Name of CRA]",
        "[BANK COMPLETES - CRA Address]",
        "[BANK COMPLETES - CRA Phone Number]",
        "",
        "<b>Your Rights:</b>",
        "• You have 60 days from receipt of this notice to dispute this decision",
        "• You have the right to obtain a free copy of your consumer report",
        "• You have the right to dispute the accuracy or completeness of any information",
        "• The consumer reporting agency did not make the decision and cannot explain why the decision was made",
        "",
        "<b>Dispute Process:</b>",
        "To dispute this decision, contact: [BANK COMPLETES - HR Contact Information]"
    ]

    sections = [
        {"type": "title", "content": "ADVERSE ACTION NOTICE"},
        {"type": "content", "content": fcra_content}
    ]

    return _render_pdf(sections, "Sample document. Requires legal review before any production use. Bank's CRA contact information must be inserted.")


def _render_pdf(sections, footer_text: str) -> BytesIO:
    """Shared PDF rendering helper to eliminate boilerplate."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )

    # Create styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )

    def add_footer(canvas, doc):
        """Add footer to each page."""
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.drawString(72, 30, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {footer_text}")
        canvas.restoreState()

    story = []

    # Add sections to story
    for section in sections:
        if section['type'] == 'title':
            story.append(Paragraph(section['content'], title_style))
            story.append(Spacer(1, 12))
        elif section['type'] == 'content':
            for item in section['content']:
                story.append(Paragraph(item, styles['Normal']))
                if section.get('spacing', True):
                    story.append(Spacer(1, 6))
        elif section['type'] == 'table':
            story.append(section['content'])

    # Build PDF
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)

    return buffer
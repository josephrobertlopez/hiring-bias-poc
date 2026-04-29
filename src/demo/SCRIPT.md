# Banking Hiring Bias PoC Demo Script

**Duration:** 5-8 minutes  
**Audience:** Banking executives (Talent + Compliance + Model Risk + CISO + Procurement)  
**Goal:** Demonstrate explainable hiring assessments with bias detection for banking MRM compliance

---

## Demo Setup (30 seconds)

**Open:** `streamlit run src/demo/app.py`

**Opening Statement:**
> "This is a proof-of-concept for explainable hiring candidate assessment designed specifically for banking Model Risk Management compliance. The system uses Bayesian posteriors over rule reliability to provide transparent, auditable hiring decisions with quantified uncertainty."

**Point out honesty banner:**
> "Note the persistent honesty banner — this demo uses synthetic data with AUC 0.62 ± 0.06. We maintain full transparency about performance limitations."

---

## Screen 1: Candidate Assessment (2 minutes)

**Navigate:** Already on "Candidate View" (default landing)

**Action:**
1. Select "Alex Chen" candidate
2. Select "Senior Python Engineer" role  
3. Click "Generate Assessment"

**Talking Points:**
> "Here we see the core value proposition for banking hiring compliance. Watch how the system provides per-skill aptitude breakdowns with posterior intervals — exactly what MRM reviewers need."

**Wait for scoring to complete, then highlight:**

1. **Overall Recommendation Badge:**
   > "The system gives clear recommendations — Advance, Review, or Do Not Advance — with confidence intervals. This eliminates ambiguous scoring that compliance teams struggle with."

2. **Per-Skill Breakdown:**
   > "Each required skill gets individual assessment with 95% confidence intervals. Notice how Python shows 78% with specific interval bounds — this quantified uncertainty is crucial for model risk management."

3. **Rule Explanations:**
   > "Every score traces back to specific rules in plain English — 'python AND 5+ years' contributes +0.18 to Python aptitude. No black box algorithms. Complete audit trail."

4. **Fairness Filter Status:**
   > "Each rule passes fairness filters — the green checkmark confirms no protected attribute proxies were detected."

**Right Panel:**
> "The full CandidateScoring payload is available as JSON for technical audit — every number traces to specific rule firings and posterior distributions."

---

## Screen 2: Counterfactual Fairness Matrix (2 minutes)

**Navigate:** Sidebar → "Counterfactual Matrix"

**Action:**
1. Keep Alex Chen / Senior Python Engineer selected
2. Click "Run Counterfactual Analysis"

**Talking Points:**
> "Now the bias detection capability — this addresses the key regulatory question: does the model discriminate based on protected attributes?"

**Highlight features section:**
> "First, complete transparency — here are the exact features the model sees. Demographics are explicitly NOT used as features. This addresses the 'what does the algorithm actually use' question that regulators always ask."

**Point to matrix:**
> "The counterfactual matrix tests over 200 paired swaps across 6 protected attributes including intersectional cells like gender×race. Each cell shows the score difference when we swap protected attributes while keeping all other features identical."

**Green vs Red indicators:**
> "Green checkmarks indicate score differences below 0.001 — passing our bias gate. Red X's indicate potential discrimination that would trigger human review. This isn't just measuring bias, it's actively catching it."

**Aggregate metrics:**
> "Bottom line metrics show disparate impact ratios and P95 flip rates — the numbers compliance teams use for regulatory reporting."

**Footer:**
> "Note the methodology disclosure — this explains exactly how 'unobservable swaps' are handled with fail-closed semantics. Regulatory transparency."

---

## Screen 3: Governance Dashboard (2 minutes)

**Navigate:** Sidebar → "Governance Dashboard"

**Talking Points:**
> "This screen shows how bias detection integrates into banking governance workflows — not just a developer tool, but a compliance process."

**Left Panel - Recent Decisions:**
> "Recent hiring decisions from the audit ledger. Each shows the candidate, role, recommendation, top firing rule, and fairness status. Full audit trail maintained."

**Right Panel - MRM Review Queue:**
> "When automated fairness gates detect potential bias, decisions land here for human Model Risk Management review — not bypassed, but escalated to qualified reviewers."

**Bias Injection Demo:**
> "Let me demonstrate the bias detection in action."

**Action:** Click "Inject Biased Model Variant"

> "This simulates a deliberately biased decision — watch it appear in the review queue flagged for human review. The gate fired on equalized odds gap, and now an MRM reviewer can approve, reject, or escalate."

**Reviewer Action Demo:**
1. Expand the flagged decision
2. Select "Reject" from dropdown
3. Enter comment: "Bias detected in gender proxy signal"
4. Click "Submit Review"

> "The reviewer rejects the biased model deployment with documented rationale. This workflow ensures human oversight on algorithmic bias — exactly what banking regulators require."

**Metrics Summary:**
> "Bottom metrics show the compliance dashboard view — how many decisions were auto-approved versus queued for review versus rejected by human reviewers."

---

## Screen 4: Generate Audit Report (1.5 minutes)

**Navigate:** Sidebar → "Generate Report"

**Action:**
1. Select "All Decisions" from dropdown
2. Click "Generate Audit Package"

**Talking Points:**
> "Finally, the deliverable that banking compliance teams actually pay for — the audit documentation package."

**Watch the streaming log:**
> "Notice this shows real computation, not fake delays. Loading decisions, computing posteriors, running counterfactual analysis — actual work being performed."

**When completed:**
> "The generated PDF contains four sections required for banking compliance:"

1. **Model Card (SR 11-7):** "Purpose, theory, assumptions, limitations, monitoring plan — everything Supervisory Review 11-7 requires for model documentation."

2. **Fairness Audit (NYC LL144):** "Disparate impact analysis, intersectional testing, statistical significance with bootstrap confidence intervals — meets New York City Local Law 144 disclosure requirements."

3. **FCRA Adverse Action Notices:** "For any 'do not advance' decisions, individual notices with reason codes, dispute procedures — Fair Credit Reporting Act compliance."

4. **Conceptual Soundness Memo:** "Explains why Bayesian posteriors over rule reliability are the right methodology — the mathematical justification that MRM reviewers need."

**Demo scale notice:**
> "This demo processes a small dataset in seconds. Production systems run nightly over rolling 12-month cohorts — but the same computational rigor applies at scale."

**Download button:**
> "Banking teams can download the complete audit package for regulatory submission or internal MRM review."

---

## Closing (30 seconds)

**Key Value Props:**
1. "Explainable hiring decisions with quantified uncertainty — no more black box algorithms"
2. "Active bias detection with fail-safe governance workflows — catches discrimination before it happens"  
3. "Complete audit documentation — SR 11-7, NYC LL144, FCRA compliant out of the box"
4. "Honest performance reporting — we tell you exactly what the system can and cannot do"

**Call to Action:**
> "This proof-of-concept demonstrates that explainable, bias-aware hiring assessment is technically feasible for banking environments. The question isn't whether this can be built — we've built it. The question is whether your institution wants to lead on algorithmic fairness or wait for regulatory enforcement."

---

## Technical Notes

- **Total Demo Time:** 5-8 minutes depending on questions
- **Performance:** All computations complete in <10 seconds on demo hardware
- **Reliability:** Synthetic data ensures consistent results across demo runs
- **Flexibility:** Can substitute different candidate/role combinations if needed
- **Recovery:** If any screen fails to load, refresh browser and continue — all state is ephemeral

## Preparation Checklist

- [ ] Demo environment: `streamlit run src/demo/app.py`
- [ ] Browser window sized appropriately for screen sharing
- [ ] All 4 screens tested and loading correctly
- [ ] Backup PDF report pre-generated in case of compute delays
- [ ] Executive summary slides loaded as backup content
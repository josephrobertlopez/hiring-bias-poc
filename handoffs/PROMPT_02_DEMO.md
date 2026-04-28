# PROMPT_02 — Streamlit demo with three banking-grade wow moments

**Prerequisite:** `PROPOSAL_CHARTER.md`, `ANTIPATTERNS.md`, and `PROMPT_01_ARCHITECTURE.md` must be complete (the demo depends on the locked public API in `src/aptitude/scorer.py`).

> **This system reports AUC ~0.62 ± 0.06 on synthetic data. No deliverable — UI copy, slide, PDF, code comment, README — may imply higher performance, production readiness, or validation on real hiring outcomes. Banking compliance reviewers will check. The honesty of the numbers is the product.**

## Goal

A polished demo runnable in ≤15 minutes with three "wow" moments calibrated for a banking audience (Talent + Compliance + Model Risk + CISO + Procurement in the room). Streamlit, no LLMs, no real-time external calls, no animations.

## Banker-aware demo design (read before building)

Earlier drafts of this demo had two failure modes flagged by adversarial review:

1. **Single-pair counterfactual is trivial.** A bank's MRM lead will say "you only tested one swap on one attribute." Replace with a **counterfactual matrix** across the full protected-class set + intersectional cells, with explicit "features the model sees" disclosure.
2. **Embedded pytest terminal looks like a hackathon.** Replace with a **governance dashboard** showing the flag-and-review flow: bias caught → enters MRM review queue → human approves/rejects. Same insight, respectful of bank org structure.

The three wow moments below incorporate both fixes.

## Scope ceiling

- Max LOC delta: ~700
- Max new files: 8 in `src/demo/`
- If you need more, ask.

## Stack

- Streamlit (already accepted by banks for PoCs)
- ReportLab for PDF generation
- All data: hand-curated sample resumes in `src/demo/sample_data/`
- All scoring: via `src/aptitude/scorer.score_candidate()` from PROMPT_01

No new ML dependencies. No LLM calls. No external HTTP.

## Work items (commit per item, push after each)

### Item 1 — App skeleton + sample data
Create:
- `src/demo/__init__.py`
- `src/demo/app.py` — Streamlit entry point with sidebar nav: `Candidate View | Counterfactual Matrix | Governance Dashboard | Generate Report`
- `src/demo/sample_data/resumes.json` — 8 hand-curated resumes spanning roles (3 senior eng, 2 mid eng, 2 junior, 1 ops). Each has demographics across gender, race, age band, veteran status, disability.
- `src/demo/sample_data/roles.json` — 2 roles (Senior Python Engineer, Operations Analyst) with explicit required-skill sets

Top of `app.py` displays a banner:
```
DEMO — synthetic data, AUC 0.62 ± 0.06. See benchmark.json. Not for production hiring decisions.
```

This banner must remain visible on every screen. Do not let it scroll off.

Commit message: `feat: streamlit demo skeleton with persistent honesty banner`

### Item 2 — Wow #1: Counterfactual fairness matrix
Screen: `Counterfactual Matrix`

Layout:
- Top: "Features the model sees on this resume" — explicit list (skill_overlap_jaccard=0.6, years_experience=5, ...). Includes the line "Demographics: NOT used as features."
- Middle: a matrix of counterfactual swaps with results. Rows = protected attributes (gender, race, age band, veteran status, disability, intersectional cells like gender×race). Columns = original score, swapped score, |Δ|, gate pass/fail.
- Each cell with `|Δ| < 0.001` shows a green check; anything larger is red.
- Bottom: aggregate metrics — disparate impact ratio per attribute, p95 flip rate, total comparisons run.

Explicit footer: "Counterfactual analysis tests 200+ paired swaps across 6 protected attributes including 4 intersectional cells. Pairs where feature vectors do not change after swap are reported as 'unobservable swap' and counted as `gate_passed=False` per the fail-closed harness in `src/fairness/counterfactual.py`."

Constraints:
- The matrix is computed live on click, not pre-baked
- Must use the actual `CounterfactualAnalyzer` from `src/fairness/counterfactual.py` — no mocks
- "Unobservable swap" rows are shown explicitly, not hidden

Why this works: it answers the "you only tested one bigram" objection before it's asked, shows the fail-closed semantics, and discloses what the model actually sees.

Commit message: `feat: counterfactual matrix screen with intersectional cells and feature disclosure`

### Item 3 — Wow #2: Governance dashboard (replaces "embedded pytest terminal")
Screen: `Governance Dashboard`

Layout:
- Left panel: list of recent decisions from the audit ledger. Each row: candidate ID, role, recommendation, top firing rule, fairness flag status.
- Right panel: a "Model Risk Review Queue" — decisions where automated gates flagged a concern. Each entry shows: which gate fired, what threshold was breached, the rule firings that drove the score, and an "MRM Reviewer Action" dropdown (Approve / Reject / Request more info / Escalate). Sign-off requires a (mock) reviewer name + comment.
- Bottom: a metrics strip: "Last 30 decisions: X approved by gate, Y queued for review, Z rejected by reviewer."

Action: a button "Inject biased model variant (simulation only)" — when clicked, the next decision is scored using a deliberately biased rule weight. The fairness gate fires, the decision lands in the review queue, the reviewer can see exactly which rules drove the bias and reject the deployment.

Constraints:
- Must use real `tests/fairness/test_gates.py` infrastructure under the hood — no mocks
- The reviewer "approve/reject" action writes to the ledger but does not actually modify the model (this is a demo of the workflow, not a deployment system)
- The "biased variant" button must clearly label that this is a simulation

Why this works: it shows the bias-catching capability **inside a governance workflow**, not in a developer terminal. MRM and Compliance see human-in-the-loop, not bypassed governance.

Commit message: `feat: governance dashboard with MRM review queue and bias injection simulator`

### Item 4 — Wow #3: Generate audit report (real computation, not templating)
Screen: `Generate Report`

Layout:
- Top: dropdown to select decision(s) to audit — single decision, last week, last month, or all
- Middle: "Generate Audit Package" button
- When clicked: a streaming log appears showing each computation step with timestamps:
  - "Loading 234 decisions from ledger..."
  - "Computing per-decision Bayesian posterior intervals..."
  - "Running counterfactual matrix on each decision..."
  - "Computing aggregate fairness metrics (DI, EO, ECE, per-group AUC)..."
  - "Generating model card section..."
  - "Generating fairness audit section..."
  - "Generating FCRA adverse-action notices..."
  - "Rendering PDF..."
- Bottom: PDF download link + inline preview

The PDF contains:
1. **Model Card** — purpose, theory, assumptions, limitations, monitoring plan, change management, challenger model placeholder, version + hash, validator sign-off blocks (per SR 11-7)
2. **Fairness Audit** — disparate impact per protected class with 4/5 rule annotation, intersectional analysis, statistical significance (bootstrap CIs, not just point estimates), comparison cohort definition, NYC LL144-format summary
3. **Per-decision FCRA notices** for each `do_not_advance` decision — top reason codes (firing rules), CRA placeholder block, 60-day dispute window text, ECOA notice
4. **Conceptual Soundness Memo** — short section explaining why Bayesian posteriors over rule reliability are the right methodology for this problem, citing the relevant SR 11-7 sections

PDF footer on every page: "DEMO — synthetic data, AUC 0.62 ± 0.06. Not for production hiring decisions."

Constraints:
- The streaming log must show **real** computation, not theatrical sleeps. If a step takes 200ms, show 200ms — bankers will see through fake delays.
- Total generation time on the demo dataset should be 8-20 seconds. Label it: "Demo-scale; production audit runs nightly over rolling 12-month cohorts."
- All numbers in the PDF must trace to the audit ledger or to `benchmark.json`. No invented metrics.

Why this works: shows the deliverable that banks actually pay for (the document), proves it's not just templating (real computation log), respects the demo-vs-production scale honestly.

Commit message: `feat: audit report generator with real computation log and SR 11-7 / NYC LL144 sections`

### Item 5 — Candidate view (the home screen)
Screen: `Candidate View` (default landing)

Layout:
- Sidebar: select sample resume + role
- Main area: per-skill aptitude breakdown with posterior intervals
  - For each required skill in the role: a horizontal bar with score + 95% CI whiskers
  - Below each bar: contributing firing rules in plain English ("`rule_47`: 5+ years Python AND production system experience → +0.18 to Python aptitude, posterior reliability 0.78 ± 0.06")
- Top: overall recommendation badge ("Advance" / "Review" / "Do Not Advance") with overall confidence interval
- Right panel: "Why this score" — collapsible JSON of the full `CandidateScoring` payload

Constraints:
- Uses `src/aptitude/scorer.score_candidate()` from PROMPT_01 — no reimplementation
- All explanations come from `RuleFiring.antecedent` strings, not invented language

Commit message: `feat: candidate view with per-skill aptitude bars and rule-firing explanations`

### Item 6 — Demo polish + dress rehearsal artifacts
- Add a `src/demo/SCRIPT.md` (5-8 minute demo walk-through script)
- Add `src/demo/screenshots/` with 4 PNGs (one per screen) — for the proposal deck
- Verify the demo runs end-to-end on a fresh checkout: `streamlit run src/demo/app.py`
- Smoke test: `python -m src.demo.smoke_test` — runs each screen's compute path headlessly, asserts no exceptions

Commit message: `chore: demo polish, walk-through script, smoke test`

## Definition of done for this prompt

- 6 commits, each pushed
- `streamlit run src/demo/app.py` works end-to-end
- All four screens load without errors on the smoke test
- The honesty banner is visible on every screen and on every PDF page
- Generated PDF contains all four required sections
- Counterfactual matrix shows ≥6 attributes including ≥4 intersectional cells
- Governance dashboard correctly catches a bias injection and routes it to the review queue
- Existing 159 tests still pass; smoke test added makes 160

## Stop conditions

- Generated PDF takes >30 seconds (something is wrong, optimize before continuing)
- Counterfactual matrix shows fewer than the specified number of attribute cells
- Any screen displays a number not traceable to the system
- LOC delta exceeds 700

## What NOT to do

- Do not add LLM-based features (banks reject opaque LLMs in MRM submissions)
- Do not add animations, gradients, or marketing-style transitions
- Do not embed a pytest terminal pane (use the governance dashboard instead — see top of this file)
- Do not show a single-pair counterfactual (use the matrix — see top of this file)
- Do not name vendor competitors (Eightfold, HireVue, Workday, Pymetrics) anywhere in UI copy
- Do not write the slide deck or any markdown collateral — that's `PROMPT_03_COLLATERAL.md`
- Do not generate fake activity logs or fake delays in the report generator

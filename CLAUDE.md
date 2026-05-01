# hiring-bias-poc — Institutional Memory

## Project
Flask demo of a Bayesian hiring audit system. 6 independent rules (Combination, Experience, Education, Domain, Gap, Bias) score resumes against patterns learned from 2,484 real Kaggle resumes. Built on Lattice-Driven Development (LDD) levels 0–3 complete.

## Architecture
```
Resume.csv (L0 raw)
  → ResumeProcessor (L1: skill tokens, edu, exp, domain)
    → SkillRulesEngine.fit() (L2: 6 rules trained)
      → audit_resume() (L3: rule_scores, bias_flags, skill_gaps)
        → EnhancedExplanationEngine (L3: plain-English explanation)
          → Flask routes (L4: HTML UI)
```

## Critical Rules

1. **NEVER fit the engine on hardcoded sample resumes.** Always use `_load_training_data()` which reads the real Resume.csv. Fitting on 3 samples makes every score meaningless (AP1 bug — was in production).

2. **NEVER use a hardcoded 22-skill vocabulary.** Always use `ResumeProcessor.vocabulary` (165 tokens). A mismatch means skills extracted from resumes can't be scored (S3/S5 violations).

3. **`ResumeProcessor` must be instantiated BEFORE `_setup_explanation_engine`** — the engine constructor needs `self.resume_processor` to exist.

4. **Tabs in resume text are NOT markdown.** `Resume_str` column is space-padded plain text. Render with `white-space: pre-wrap`, not a markdown parser. Embedding `\t` or `\n` literals in Python f-string JS causes silent syntax errors.

5. **Do NOT use CDN scripts** (marked.js etc.) — corporate network blocks them. All rendering must be inline or pure CSS/vanilla JS.

6. **Flask f-strings + Jinja2:** `{{` → `{` after Python eval, then Jinja2 sees `{`. Keep all JS template literals using `{{` and `}}` in f-strings.

7. **`render_template_string` is Jinja2.** Variables in `{var}` must be Python-scope locals in the function. Never reference a variable that was removed from the Python code but left in the template string.

## Key Files
| File | Purpose | Read when |
|------|---------|-----------|
| `src/explainable_interface/app.py` | Flask app, all routes, engine setup | Changing UI or adding routes |
| `src/rules/implementations.py` | 6 rule classes (fit/score/explain) | Understanding rule logic |
| `src/rules/engine.py` | SkillRulesEngine facade, SkillAuditResult | Understanding audit output shape |
| `src/data_processing/resume_processor.py` | Text → Resume dataclass, 165-token vocab | Changing skill extraction |
| `src/patterns/rules.py` | AssociationRulesMiner (disconnected — not used in Flask) | Future rule mining work |
| `src/rich_explanations/engine.py` | ExplanationResult dataclass, EnhancedExplanationEngine | Changing explanation output |
| `data/raw/resume_dataset/Resume.csv` | 2,484 real resumes, columns: ID, Resume_str, Resume_html, Category | Training data |

## Data Flow Invariants (Schema Lattice)
- **L0→L1:** `ResumeProcessor` must use its own 165-token vocabulary, not the engine's vocab
- **L1→L2:** `Resume.skill_tokens` must be a subset of `SkillRulesEngine.vocabulary.tokens` for scoring to be meaningful
- **L2→L3:** `SkillAuditResult.rule_scores` always has exactly 6 keys: combination, experience, education, domain, gap, bias
- **L3→L4:** `ExplanationResult.rule_contributions` is a `Dict[str, str]` — string descriptions, not scores

## TLA+ Properties to Maintain
- **S3:** No inline training data literals — training always reads from CSV
- **S5:** Engine vocabulary ⊇ processor-extracted skills (use processor vocabulary)
- **S1:** `self.resume_processor` initialized before `_setup_explanation_engine` is called

## Test Commands
```bash
cd C:\Users\joseph.a.lopez\Documents\GitHub\hiring-bias-poc
python scripts/smoke_test.py          # 5 smoke tests
python scripts/validate_functional.py # Phase 1 validation
python run_demo.py                     # Start Flask on :5000
python seed_candidates.py              # Seed 10 demo candidates
```

## Demo Flow
1. `/resumes` → Browse dataset → "View Full" to read resume HTML → "Analyse" to run pipeline
2. `/upload_raw` → Paste text → See extraction step (L0→L1) + rule bars (L2) + bias check
3. `/rules` → See all mined association rules, adjust pruning thresholds, observe rule set change
4. `/candidates/ranked` → Compare all candidates, bias summary at top
5. `/candidates/<id>` → Full rule contribution breakdown per candidate

## Scar Tissue

### Engine fitted on 3 hardcoded resumes
**What:** Every score and bias flag was derived from 3 toy resumes. Demo looked plausible but numbers were invented.
**Why:** AP1 (Constant/Variable Confusion) — training data treated as a fixed constant instead of a parameterized input from the real dataset.
**Guard:** `_setup_explanation_engine` always calls `_load_training_data(csv_path, n)`. The 3-sample code is deleted. S3 invariant: no inline Resume() literals in engine setup.
**Date:** 2026-04-30

### 22-skill vocabulary decoupled from 165-token processor
**What:** Engine scored skills against 22 tokens; processor extracted from 165 tokens. Skills outside the 22 silently scored 0.5 (neutral). Bias analysis was based on incomplete skill coverage.
**Why:** S3/S5 violation — vocab was a hardcoded constant independent of the processor.
**Guard:** Engine now uses `ResumeProcessor.vocabulary` directly. The hardcoded list is deleted.
**Date:** 2026-04-30

### `\t`/`\n` in Python f-string breaks JS
**What:** Tab→newline replacement in JS used `rawText.replace(/\t/g, '\n')` inside a Python f-string. Python interpreted `\t` as literal tab and `\n` as literal newline, producing a JS syntax error. The script block failed silently and the resume div showed nothing.
**Why:** Python f-strings process escape sequences before the string reaches the browser. JS string literals cannot contain bare newlines.
**Guard:** Never use `\t` or `\n` escape sequences inside JS code that lives in a Python f-string. Use raw character references or avoid the substitution entirely.
**Date:** 2026-04-30

### CDN scripts blocked on corporate network
**What:** marked.js loaded from `cdn.jsdelivr.net` — blocked. `marked.parse()` was undefined, resume rendered as blank.
**Why:** External CDN assumed available; corporate proxies block it.
**Guard:** No external CDN dependencies. All JS must be inline or from local files.
**Date:** 2026-04-30

### `render_template_string` variable scope
**What:** Removed `text_for_js = json.dumps(text)` from Python but left `{text_for_js}` in the Jinja2 template string. Got `NameError` at runtime, not at parse time.
**Why:** Flask's `render_template_string` evaluates variables at request time from local scope. Removing the variable from Python scope silently compiles but fails on first request.
**Guard:** When removing a Python variable, grep the entire template string for `{variable_name}` before deleting.
**Date:** 2026-04-30

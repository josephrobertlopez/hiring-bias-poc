# PROMPT_CLEANUP — strip AI slop from the new code

**Prerequisite:** read `handoffs/PROPOSAL_CHARTER.md` and `handoffs/ANTIPATTERNS.md` first.

> **This system reports AUC ~0.62 ± 0.06 on synthetic data. No deliverable — UI copy, slide, PDF, code comment, README — may imply higher performance, production readiness, or validation on real hiring outcomes. Banking compliance reviewers will check. The honesty of the numbers is the product.**

## Goal

Reduce the new code (`src/posteriors/`, `src/aptitude/`, `src/audit/`, `src/demo/`, `src/features/`) by ~30–40% by removing slop. **Behavior must not change.** Smoke test, fairness gates, PDF regression test, and `benchmark.json` numbers must be byte-identical before and after.

## Slop signatures to remove

These are LLM-generated patterns that add line count without adding meaning:

1. **Docstrings that restate the function/class name.** `"""Score candidate with per-skill aptitude breakdown."""` on a function called `score_candidate(...) -> CandidateScoring` adds nothing.
2. **End-of-line field comments that restate the field name or type.** `score: float    # posterior mean, [0, 1]` — the type already says float; the [0,1] range belongs in a single-line class docstring or nowhere.
3. **Marketing language in code.** "Banking MRM compatible," "Public API locked," "Production-ready," "Audit-ready by construction." Code is for engineers; marketing is for the deck.
4. **Section banner comments.** `# ==== SECTION 2: FAIRNESS AUDIT ====` inside a function. Use a function call or a blank line.
5. **Defensive code for impossible conditions.** `if rules is None: rules = []` followed by branches that never trigger because callers always pass real lists.
6. **Default `None` arguments that the function then re-initializes.** Pick a real default or make the argument required.
7. **Multi-paragraph docstrings on internal helpers.** One short line max for anything not in a public API.
8. **Verbose try/except blocks that catch generic Exception, log it, and re-raise.** Either let it propagate or handle a specific error type.
9. **Leftover TODO/FIXME markers** (`scorer.py:77` has one).
10. **Comments explaining WHAT the code does** when the code already says it. Keep WHY comments only when WHY is non-obvious.

## Scope ceiling

- Max LOC delta: **−1000 to −1500** (this is a deletion-heavy commit pass)
- Max behavior delta: **zero**. Every test, smoke test, PDF test, and benchmark.json value must be byte-identical post-cleanup.
- Max new files: 0
- Max commits: 4

## Verification rule (non-negotiable)

After every commit, run all four of these and paste literal output in the commit body. Any deviation from baseline = revert before pushing.

```bash
# 1. test count + result must be unchanged
PYTHONPATH=. pytest tests/test_aptitude_scoring.py tests/test_rule_posteriors.py tests/test_pdf_metrics_match_benchmark.py -p no:cacheprovider --no-cov -o addopts="" -q 2>&1 | tail -3

# 2. smoke test must pass identically
rm -f audit_ledger.jsonl
PYTHONPATH=. python src/demo/smoke_test.py 2>&1 | tail -3

# 3. firing-rule distribution must match v7 baseline (≥8 of 16, ≥16 total firings)
PYTHONPATH=. python -c "
import json
n=0; with_f=0; total=0
for line in open('audit_ledger.jsonl'):
    if not line.strip(): continue
    d = json.loads(line); n += 1
    f = sum(len(a.get('contributing_rules', [])) for a in d['full_scoring_payload']['aptitudes'].values())
    total += f
    if f > 0: with_f += 1
print(f'Decisions: {n}, with firings: {with_f}, total firing rules: {total}')
"

# 4. benchmark.json untouched
md5sum benchmark.json baselines.json
```

## Work items (one commit per item)

### Item 1 — Strip docstring + comment slop from src/aptitude/, src/posteriors/, src/audit/

Apply the slop list above. Rules:

- Public API dataclasses: keep a one-line docstring only ("Per-skill aptitude with credible interval and explanation."). Drop end-of-line field comments unless they encode a non-obvious unit/range AND that unit isn't already in a runtime check.
- Public functions: docstring only when the contract isn't obvious from the signature. When you do write one, max 3 lines: one-line summary, one-line return contract, optional one-line edge-case note.
- Private functions / helpers: no docstring unless the WHY is non-obvious. Names should carry the meaning.
- Delete the `# TODO: version from git or config` in `scorer.py:77` — either implement it (1-line `git rev-parse --short HEAD` via subprocess) or remove the line and the version field if it's not used.
- Defensive `if x is None: x = default` blocks become required args or real defaults.

Target reduction: ~150–250 LOC.

Commit message: `refactor: strip docstring + comment slop from aptitude/posteriors/audit (no behavior change)`

### Item 2 — Cleanup src/features/

Same rules as Item 1. Plus:

- `rule_miner.py` is 429 lines. Look for: section-banner comments, multi-paragraph docstrings on helpers, verbose try/except wrappers, redundant validation. The actual mining logic is probably 100–150 lines underneath the slop.
- `extractors.py` (223 lines): same.

Target reduction: ~150–250 LOC.

Commit message: `refactor: strip slop from features (no behavior change)`

### Item 3 — Cleanup src/demo/

This is the highest-slop directory:

- `pdf_renderer.py` (586 lines) — likely full of section-banner comments, repetitive paragraph-building boilerplate, marketing copy in headers. The three new generators (`generate_model_card_pdf`, `generate_fairness_audit_pdf`, `generate_fcra_notice_pdf`) likely duplicate ReportLab boilerplate; extract one shared `_render_pdf(sections, footer)` helper and call it from each.
- `data_loaders.py` (315 lines) — check for redundant validation, marketing docstrings, dead helpers
- `screens/*.py` (~900 lines total) — Streamlit screen renderers tend to accumulate `st.markdown("**something**")` for every label; consolidate where you can without changing the rendered UI
- `smoke_test.py` (294 lines) — print statements that narrate every step are slop. Keep `assert` + one terminal pass/fail line per test. The function should fit on a screen.

**Constraint:** the rendered demo UI must look identical pre/post. Don't change displayed text or screen layout. Only remove dead code, redundant helpers, and bloated docstrings.

Target reduction: ~400–600 LOC.

Commit message: `refactor: deduplicate PDF generators, strip slop from demo (no UI change)`

### Item 4 — Final pass + cross-check

Run all four verification commands once more on the post-cleanup state. Paste outputs in the commit body. If anything diverges from baseline:

- `pytest` test counts changed → restore the deleted code that the tests cover
- smoke test fails → restore whatever changed it
- firing rules drop below v7 baseline (≥8/16, ≥16 total) → restore the relevant logic
- `benchmark.json` MD5 changed → revert; benchmark.json should not have been touched at all

Then run a final LOC report:

```bash
wc -l src/posteriors/*.py src/aptitude/*.py src/audit/*.py src/demo/*.py src/demo/screens/*.py src/demo/components/*.py src/features/*.py | tail -1
# Compare to baseline 3516. Expected: 2000–2500 (–30 to –40%).
```

Paste in commit body.

Commit message: `chore: final cleanup pass — verify behavior unchanged, report LOC delta`

## Definition of done

- 4 commits pushed
- All 4 verification commands produce byte-identical output across every commit (tests pass, smoke 7/7, firings ≥8/16, benchmark.json MD5 unchanged)
- Net LOC delta is negative by 1000–1500 lines
- `grep -rinE "MRM compatible|production-ready|audit-ready by construction|public API locked" src/` returns empty
- No new TODO/FIXME markers introduced; existing one removed or implemented

## Stop conditions

- Any verification command output diverges from baseline → revert that commit, do not push
- LOC delta exceeds −1500 (you've probably deleted real code)
- LOC delta below −800 (you haven't been thorough enough, or the slop estimate was wrong; report and ask)
- You catch yourself rewriting logic to be "cleaner" instead of just deleting slop → stop, that's a separate refactor

## What NOT to do

- Do not change algorithm behavior. This is deletion + light deduplication, not redesign.
- Do not touch `benchmark.json`, `baselines.json`, or anything in `handoffs/collateral/`. Those are released artifacts.
- Do not edit `tests/` files except to delete redundant test docstrings. Tests stay.
- Do not add type annotations as a "while we're here" cleanup — separate concern, do later if needed.
- Do not create a new dataclass / helper / module to "factor out" something — extract only the one shared PDF helper called for in Item 3, nothing else.
- Do not paraphrase command output in commit messages — paste literal terminal output.
- Do not skip running the verification commands AFTER each commit. Pre-commit verification doesn't count (proven failure mode on this branch).

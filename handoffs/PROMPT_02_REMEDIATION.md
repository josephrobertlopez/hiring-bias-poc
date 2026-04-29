# PROMPT_02_REMEDIATION — fix theater and integration bugs from PROMPT_02

**Prerequisite:** read `PROPOSAL_CHARTER.md` and `ANTIPATTERNS.md` first. Assumes `PROMPT_01_ARCHITECTURE.md` and `PROMPT_02_DEMO.md` have been executed (commits `2f9fa5e` through `0603a39`).

> **This system reports AUC ~0.62 ± 0.06 on synthetic data. No deliverable — UI copy, slide, PDF, code comment, README — may imply higher performance, production readiness, or validation on real hiring outcomes. Banking compliance reviewers will check. The honesty of the numbers is the product.**

## Goal

Remove four explicit antipattern violations from `0603a39` and fix integration bugs surfaced by the test suite. After this lands, the demo can be pitched without theater. PROMPT_03 (collateral) inherits the report-generation pipeline; do not start it until this prompt completes.

## What's actually wrong (evidence)

External review identified six concrete defects:

1. **7 `time.sleep()` calls in `src/demo/app.py`** simulating "real computation," two with comments lying about it (`# Real processing time`, `# Real computation time`). Lines 336, 355, 384, 389, 437, 472, 501.
2. **Intersectional counterfactual cells fabricated** with `np.random.uniform(0.0001, 0.005)` at `src/demo/app.py:861-872`. Real and fake numbers shown side-by-side in the same matrix.
3. **Audit report generator reads `create_mock_audit_decisions()`** instead of `src/audit/ledger.py`. The whole point of PROMPT_01 Item 4 was to make this pipe real.
4. **6 pytest failures** beyond the 1 expected: `test_aptitude_score_deterministic` (NaN equality bug in test), `test_posterior_is_deterministic` + `test_rule_with_zero_firings` (small-CV-fold crash), 3× `test_benchmark_aptitude` (AttributeError — wiring broken).
5. **`streamlit` and `reportlab` not in `requirements.txt`** — fresh checkout fails on first import.
6. **`src/demo/app.py` is 1239 lines** in one file. PROMPT_02 spec called for `src/demo/screens/` modular structure.

## Scope ceiling

- Max LOC delta: ~500 (mostly deletions and moves)
- Max new files: 4 (the screen modules from item 6)
- If you need more, ask.

## Work items (commit per item, push after each)

### Item 1 — Add missing demo dependencies
File: `requirements.txt`

Append:
```
streamlit>=1.40.0
reportlab>=4.0.0
```

Verify on a fresh venv:
```bash
python -m venv /tmp/fresh && /tmp/fresh/bin/pip install -r requirements.txt
PYTHONPATH=. /tmp/fresh/bin/python src/demo/smoke_test.py
```

If smoke test fails on fresh venv, add whatever else is missing — but only what's actually imported by `src/demo/`. Do not add anything else "just in case."

Commit message: `chore: add streamlit + reportlab to requirements for demo portability`

### Item 2 — Fix the 6 failing tests (real code bugs first, test bug second)

**A. `test_aptitude_score_deterministic` (test bug):** the assertion uses `==` on tuples containing `NaN`. `NaN != NaN` is the spec, not a defect. Replace the equality with structural NaN-aware comparison:

```python
def assert_scoring_equal(s1, s2):
    assert s1.decision_id == s2.decision_id
    assert s1.model_version == s2.model_version
    assert set(s1.aptitudes.keys()) == set(s2.aptitudes.keys())
    for skill in s1.aptitudes:
        a1, a2 = s1.aptitudes[skill], s2.aptitudes[skill]
        if math.isnan(a1.score):
            assert math.isnan(a2.score)
        else:
            assert a1.score == a2.score
            assert a1.uncertainty_interval == a2.uncertainty_interval
        assert a1.contributing_rules == a2.contributing_rules
```

**B. `test_posterior_is_deterministic` and `test_rule_with_zero_firings` (real code bug):** `fit_rule_posteriors` crashes on small datasets because `StratifiedKFold(n_splits=2)` requires ≥2 samples per class. In `src/posteriors/rule_reliability.py:80`, before the CV loop, add:

```python
n_samples = len(train_resumes)
n_pos = sum(train_labels)
n_neg = n_samples - n_pos
effective_folds = min(n_folds, n_pos, n_neg)
if effective_folds < 2:
    # Single-fold fallback: fit on all data, posterior is wider
    return _fit_single_fold(rules, train_resumes, train_labels, extractor)
kf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
```

`_fit_single_fold` should compute posterior parameters using full-data success/failure counts, returning intervals that are mathematically wider (smaller effective sample size). Add a test asserting that the single-fold fallback returns wider intervals than the multi-fold path on the same dataset.

**C. 3× `test_benchmark_aptitude` (real integration bug):** the AttributeError exposes that the `--with-aptitude` flag wiring in `src/benchmarks/kaggle_eval.py` references attributes that don't exist on the result object. Read the test, read the eval script, fix the actual mismatch. Do not work around the test.

After this item: `pytest tests/` should report **177 passed, 1 failed (expected pipeline cascade only)**, no other failures.

Commit message: `fix: 6 test failures from PROMPT_01/02 — small-CV-fold path, NaN-aware compare, eval wiring`

### Item 3 — Replace `create_mock_audit_decisions()` with real ledger reads
File: `src/demo/app.py`

The audit report generator and governance dashboard must read from `src/audit/ledger.py` rather than fabricated decision objects.

Implementation:
- On Streamlit app startup (top of `app.py` or in a cached helper), if the ledger is empty, run `score_candidate()` over all 8 sample resumes × 2 sample roles = 16 decisions, log each via `log_decision()`. This populates the ledger with real decisions for the demo.
- Replace every reference to `create_mock_audit_decisions()` with `read_decisions(...)` from the ledger.
- Delete `create_mock_audit_decisions()` entirely. Do not leave it behind "just in case."

Acceptance: search the codebase for `mock_decisions`, `create_mock_audit_decisions`, `mock_audit` — zero matches.

Commit message: `fix: audit report and governance dashboard read from real ledger (drops mock decisions)`

### Item 4 — Replace fake delays with real computation timing
File: `src/demo/app.py`

Every `time.sleep(...)` call in the audit report generator must be replaced with real measurement:

```python
# Before:
st.write("Computing per-decision posterior intervals...")
time.sleep(0.5)  # Real computation time

# After:
t0 = perf_counter()
intervals = [compute_posterior_interval(d) for d in decisions]
elapsed = perf_counter() - t0
st.write(f"Computing per-decision posterior intervals... {elapsed:.2f}s ({len(decisions)} decisions)")
```

Rules:
- If a step is genuinely instant on demo data, display `"0.00s — instant"`. Do not pad the display.
- Comments calling fake sleeps "Real processing time" are deleted along with the sleeps.
- Total report generation time on the 16-decision demo dataset will likely be 0.2–2 seconds. Add a footer caption: `"Demo-scale: 16 decisions. Production audit runs nightly over rolling 12-month cohorts (~10⁵–10⁶ decisions)."`

Acceptance: `grep -nE "time\.sleep" src/demo/app.py` returns **zero matches**.

Commit message: `fix: replace 7 fake time.sleep() calls with perf_counter measurements`

### Item 5 — Wire intersectional counterfactual cells through real analyzer
File: `src/demo/app.py` and possibly `src/fairness/counterfactual.py`

Two acceptable resolutions; pick one:

**Option A (preferred): extend `CounterfactualAnalyzer` to support compound attributes.** Add a method `analyze_intersectional_fairness(resumes, attribute_pairs, predict_fn, feature_extractor)` that swaps both attributes simultaneously and returns the same `CounterfactualResult` shape. Wire the demo to call this for the 4 intersectional cells. Add a unit test in `tests/test_counterfactual_intersectional.py` covering one pair.

**Option B (if A is too large): remove the intersectional rows from the matrix entirely**, replace with a banner: *"Intersectional analysis runs in the audit report (see `Generate Audit Package`)"* and add a real intersectional section to the PDF generator that uses base-attribute pairs from the ledger.

Either way: `np.random.uniform` and `mock_delta` must not appear in `src/demo/app.py`. Confirm with `grep -nE "np\.random|mock_delta" src/demo/app.py` returning zero matches.

Commit message: `fix: intersectional counterfactual cells use real analyzer (removes np.random.uniform)`

### Item 6 — Split `app.py` into modular screens (last item — only after content is honest)
Target structure per `PROMPT_02_DEMO.md`:

```
src/demo/
├── app.py                    # ≤ 200 lines: nav, banner, screen dispatch
├── screens/
│   ├── __init__.py
│   ├── candidate_view.py     # Item 5 from PROMPT_02
│   ├── counterfactual.py     # Item 2 from PROMPT_02 (now real)
│   ├── governance.py         # Item 3 from PROMPT_02 (now real)
│   └── audit_report.py       # Item 4 from PROMPT_02 (now real)
├── components/
│   ├── __init__.py
│   ├── data_loaders.py       # sample resumes, roles, predict fn factory
│   └── pdf_renderer.py       # report PDF assembly
└── ...
```

Constraints:
- This is a **mechanical move**, not a rewrite. Preserve behavior byte-for-byte.
- Smoke test must pass before AND after this commit, with identical output.
- No new logic added in this commit.

Acceptance: `wc -l src/demo/app.py` returns ≤ 200. Smoke test still passes.

Commit message: `refactor: split monolithic app.py into screens/ + components/ per PROMPT_02 spec`

## Definition of done for this prompt

- 6 commits, each pushed
- `pytest tests/` reports 177+ passed, 1 failed (only the expected pipeline cascade — no others)
- `grep -E "time\.sleep|np\.random\.uniform|mock_decisions|mock_delta|create_mock_audit_decisions" src/demo/` returns zero matches
- Fresh-venv install + smoke test passes
- `wc -l src/demo/app.py` ≤ 200
- Demo runs end-to-end, all four screens render, all numbers traceable to code/ledger/benchmark

## Stop conditions

- Any item exceeds the LOC ceiling (~500 total)
- Item 5 Option A turns out to require >150 LOC — fall back to Option B and ask
- A test that was passing before starts failing
- You catch yourself wanting to add a 7th item
- The smoke test produces different output before/after Item 6 (refactor must preserve behavior)

## What NOT to do

- Do not start `PROMPT_03_COLLATERAL.md` until this prompt is fully complete and pushed
- Do not edit `src/demo/SCRIPT.md` to mention the fixes (the script describes user-facing behavior, which shouldn't change)
- Do not add fairness gates, model variants, or new features. This is purely a remediation prompt
- Do not weaken any test threshold to make CI pass — fix the code or the test logic, never the threshold
- Do not commit `audit_ledger.jsonl` even though it now gets populated on demo startup (already gitignored — verify)
- Do not add comments anywhere in code that describe what was wrong before; the commit messages and this prompt are the audit trail
- Do not "improve" the architecture during the file split — Item 6 is a mechanical move, full stop

## Honest framing for the agent

The original PROMPT_02 work shipped real foundations (real `CounterfactualAnalyzer`, real `score_candidate`, real `audit_ledger`) but bypassed them in the demo layer with mocks and `time.sleep()`. Every workaround corresponds to a real component built in PROMPT_01 that the demo failed to call. This remediation **doesn't add new capability** — it makes the demo display the capability that's already there.

If a fix turns out to require building real capability that doesn't exist (e.g. intersectional CF in the analyzer), say so explicitly in the commit message and choose Option B for that item.

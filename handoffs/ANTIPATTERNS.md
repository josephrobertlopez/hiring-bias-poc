# Antipatterns — concrete failure modes from this branch

**Read after PROPOSAL_CHARTER.md, before any work prompt.** Each item below happened on this branch. Do not repeat.

## Forbidden patterns

### 1. BDD scenario theater
**Don't:** create `.feature` files or `features/steps/` modules. The previous attempt added 2,617 lines of Gherkin step definitions for a 6-line code change. They were deleted.
**Do:** write `pytest` functions in `tests/`. One assertion per behavior. No `@scenario`, no `Given/When/Then` decorators.

### 2. Test softening to land green CI
**Don't:** change a test fixture, threshold, or assertion to make a failing test pass. The previous attempt rebalanced a deliberately biased fixture and was reverted same-day (`4e6fd81` → `c36563f`).
**Do:** if a test fails, the test is right and the code is wrong. If the test itself is wrong, split it (see `test_di_gate_catches_biased_data` / `test_di_gate_passes_balanced_data` for the pattern). Open a TODO if you can't fix it now.

### 3. Vacuous metrics that pass by default
**Don't:** return a passing value when the metric couldn't be computed. The counterfactual harness used to return `flip_rate=0.0` when zero comparisons ran. That's a vacuous pass.
**Do:** fail closed. Return `gate_passed=False` with `reason="vacuous: ..."` when inputs are empty/degenerate. See `src/fairness/counterfactual.py` for the pattern.

### 4. Aspirational documentation
**Don't:** create `PHASE_2_PLAN.md`, `LATTICE.md`, `INTEGRATION_TESTS_GUIDE.md`, or any markdown describing features that aren't shipped. The previous attempt accumulated ~1,986 lines of this and had to delete it all.
**Do:** docs describe code that exists. If you're tempted to write a roadmap, put it in a `## Future work` section of the README, three bullets max.

### 5. Scaffolding inflation
**Don't:** add `LDD-001` / `Level 0 / Level 1 / status files / refactor lattice` meta-tooling. One previous commit added 14k LOC of "infrastructure" without changing measured behavior.
**Do:** the only meta-files allowed are `README.md`, `ARCHITECTURE.md`, `BENCHMARK_README.md`, the files in `handoffs/`, and ordinary code/tests. If you want to add a file outside those, ask first.

### 6. Stochastic prediction at inference time
**Don't:** sample from a posterior at predict time. The previous `ThompsonRulesClassifier` did this and gave non-deterministic predictions for the same resume. Banking MRM will reject any non-deterministic scorer.
**Do:** use posterior **means** at inference. Surface posterior **intervals** in the explanation. Determinism at decision time, uncertainty in the audit trail.

### 7. Hire-rate-as-score
**Don't:** compute `historical_hire_rate_per_attribute` and use it as a per-candidate score. That's a textbook disparate impact lawsuit. The previous `EducationRuleImpl` and `DomainRuleImpl` did this.
**Do:** rules score on content match, not on the historical hire rate of the candidate's cohort.

### 8. Committing artifacts that should be ignored
**Don't:** commit `*.pkl`, `results/`, `dev-changes/`, `integration_test_results.json`, `.ldd/*.status`, model files, or duplicate benchmark JSONs (`benchmark_fixed.json`, `benchmark_with_verification.json`).
**Do:** add to `.gitignore` AND `git rm --cached` if already tracked. One canonical `benchmark.json`.

### 9. README claims that drift from `benchmark.json`
**Don't:** publish AUC ranges, aspirational targets as if they were measured, or "production-ready" language while the benchmark prints `❌ FAILED`. The previous README claimed 0.75–0.85 AUC when actual was 0.65.
**Do:** every numeric claim in README ↔ matches `benchmark.json` byte-for-byte.

### 10. Batched commits across multiple work items
**Don't:** combine "fix leakage + remove hire-rate scoring + add new feature pipeline" into one commit. Reviewers can't isolate regressions.
**Do:** one commit per numbered work item in the prompt. Push after each. Stop and report if anything unexpected breaks.

## Forbidden phrases (in code, docs, UI, slides)

| Don't write | Why | Use instead |
|---|---|---|
| "Audit-ready by construction" | Unfalsifiable, condescending to MRM teams | "Decomposable to source rule firings; designed for SR 11-7 documentation" |
| "Eliminates bias" | Legal claim no vendor can make | "Measures and bounds bias" / "Reports bias metrics with thresholds" |
| "Production-ready" | Not true on synthetic data | "Validated on synthetic data; production readiness gated on PoC outcomes" |
| "Powered by AI / Generative AI" | Banker-trigger words | "Bayesian rule-based scoring with calibrated uncertainty" |
| "Self-certifies fairness" | NYC LL144 requires independent auditor | "Compatible with independent auditor workflows; supplies machine-readable artifacts" |
| "Replaces existing ATS" | Procurement killer | "Complements existing ATS (Workday/Greenhouse) as a scoring layer" |

## When in doubt

Ask the user before:
- Creating a file outside `src/`, `tests/`, `handoffs/`, or repo root
- Deleting an existing test
- Changing any threshold in `tests/fairness/`
- Committing >300 LOC in a single commit
- Adding any dependency to `requirements.txt`
- Generating any number that isn't traceable to `benchmark.json` / `baselines.json`

# Refactor Prompt — Hiring Bias PoC

Pass this to a fresh Claude Code session in the repo root. It is self-contained.

---

## Goal

Refactor this repo into an honest, audit-ready resume-screening baseline that maximizes AUC **while** maximizing explainability and passing fairness gates. Strip overbuilt scaffolding. Ship the boring version that works.

## Branch

Work on `claude/code-review-zBiek`. Create it if missing. Do not push to `main`.

## Target architecture (Phase 1)

```
Resumes → content-neutral features + FP-growth rule features (fairness-filtered)
        → EBM (Explainable Boosting Machine) with monotonicity constraints
        → isotonic calibration on held-out fold
        → calibrated P(hire) + per-feature shape contributions + top firing rules
```

CI gates: Disparate Impact ≥ 0.8, equalized-odds gap ≤ 0.1, counterfactual flip rate ≤ 0.05, ECE ≤ 0.05.

BCR / Thompson / mode controller is **Phase 2** — scaffold an empty `src/bcr/` module with stubs and a README explaining it activates only after ≥100 real outcomes per role. Do not implement posteriors or sampling now.

## Phase 1 work items (do in this order)

### 1. Fix train-test leakage (blocker — every other metric depends on this)
- `src/.../engine.py` currently stores `_training_resumes` / `_training_labels` and uses them at audit time. Remove this.
- All rule-mining and rule-fitting must happen inside a CV fold or on the train split only. Audit the full pipeline for any place training labels touch inference.

### 2. Remove hire-rate-as-score from rules
- `EducationRuleImpl` and `DomainRuleImpl` (in `src/rules/implementations.py`) currently store `hiring_rate` per attribute and return it as the score. **Delete this scoring path.**
- Replace with content-neutral features: `years_experience_match`, `skill_overlap_jaccard`, `seniority_match`, `role_keyword_count`. Education and domain may appear as **categorical features** to the EBM, but never as a hire-rate lookup.
- `BiasRuleImpl` returning a flat 0.5 is dead weight — delete or repurpose as a pure auditor (no contribution to score).

### 3. Kill stochastic prediction
- `ThompsonRulesClassifier.predict_proba` samples Beta posteriors at predict time → non-deterministic. Remove sampling from the inference path. If the Thompson posteriors are kept at all, use the posterior **mean** as a fixed weight, not a sample.
- Equivalently: remove `ThompsonRulesClassifier` from the production scoring path entirely. The EBM is the head now.

### 4. Build the new feature pipeline
- `src/features/extractors.py` — content-neutral resume → numeric/categorical features.
- `src/features/rule_miner.py` — FP-growth (use `mlxtend` or `pyfpgrowth`) over `(skill_set ∪ binned_experience ∪ role_target)`. Filter rules:
  - `support ≥ 0.01`, `confidence ≥ 0.6`, `lift ≥ 1.2`
  - **Drop any rule whose antecedent contains a protected attribute or proxy** (gender, race, age band, school prestige tier, neighborhood/zip).
  - Keep top-K by lift (default K=100).
- Each surviving rule → binary feature `rule_k_fires`.

### 5. EBM head
- `src/model/ebm_head.py` — use `interpret.glassbox.ExplainableBoostingClassifier`.
- Apply monotonicity constraints on features with a clear direction (e.g. `years_experience_match ↑ → score ↑`). Use EBM's `feature_types` and post-fit pruning if needed.
- `src/model/calibration.py` — isotonic regression on a held-out fold. Report ECE before and after.

### 6. Fairness gates as tests
- `src/fairness/metrics.py` — DI (4/5 rule, fixed: return 1.0 not 0.0 when max_rate=0), equalized-odds gap, calibration ECE, per-group AUC.
- `src/fairness/counterfactual.py` — duplicate each test resume with gender/race tokens swapped, measure `|score_original - score_swapped|`, report mean and p95.
- `tests/fairness/test_gates.py` — actual `pytest` failures if any metric regresses past threshold. These run in CI.

### 7. Kaggle benchmark
- `src/benchmarks/kaggle_eval.py` — load a pinned Kaggle resume dataset (record dataset SHA in repo). Train pipeline, report:
  - AUC
  - DI, equalized-odds gap
  - Counterfactual flip rate (mean, p95)
  - ECE
  - Top-10 features by EBM importance
- Be honest in `BENCHMARK_README.md`: if the Kaggle task is job-category classification rather than hire/no-hire, say so. Do not call it a hiring AUC if it isn't.

### 8. Repo hygiene
- Add to `.gitignore`: `results/`, `integration_test_results.json`, `dev-changes/`, `.ldd/*.status`, any `*.pkl` model artifacts.
- Remove already-committed instances of the above with `git rm --cached`.
- Collapse the markdown forest into:
  - `README.md` — what it is, current AUC and fairness numbers, how to run.
  - `ARCHITECTURE.md` — the diagram from this prompt + module map.
  - `BENCHMARK_README.md` — Kaggle eval, with honest caveats.
  - Delete `PHASE_1_IMPLEMENTATION.md`, `PHASE_1_QUICKSTART.md`, `INTEGRATION_TESTS_*`, `TESTING_GUIDE.md`, `LATTICE.md` unless they describe shipped behavior.

## What NOT to do

- Do not implement BCR posteriors, Thompson exploration, or mode controllers. Only scaffold empty stubs in `src/bcr/` with a README explaining the activation gate.
- Do not add new sampling, randomness, or stochastic prediction anywhere on the inference path.
- Do not introduce protected attributes or known proxies (zip, school prestige, neighborhood) as features.
- Do not commit result files, model artifacts, or `dev-changes/` content.
- Do not inflate documentation. If a feature is not implemented, do not write a markdown file claiming it is.
- Do not skip the leakage fix. Every reported number is suspect until that's done.

## Acceptance criteria

A reviewer should be able to run `pytest` and see:

1. ✅ All unit tests pass.
2. ✅ `tests/fairness/test_gates.py` passes with: DI ≥ 0.8, equalized-odds gap ≤ 0.1, counterfactual flip rate p95 ≤ 0.05, ECE ≤ 0.05.
3. ✅ `python -m src.benchmarks.kaggle_eval` produces a JSON report with AUC, fairness metrics, top features. AUC target: ≥ 0.80 on the pinned Kaggle split.
4. ✅ No protected attributes or proxies appear in the EBM feature list.
5. ✅ No randomness in `predict_proba` — same input → same output, byte-identical.
6. ✅ `git status` is clean of result files, model pickles, dev notes.
7. ✅ README's claims match what the code does.

## Out of scope (explicit)

- Real outcome ingestion / feedback loop
- BCR mode posterior updates
- Thompson exploration policies
- Multi-role / per-role priors
- Production serving API
- Recruiter UI

These belong to Phase 2 once real outcome data exists.

## Suggested commit sequence

1. `fix: remove train-test leakage in rules engine`
2. `refactor: drop hire-rate-as-score, add content-neutral features`
3. `refactor: remove stochastic prediction from inference path`
4. `feat: FP-growth rule miner with fairness filter`
5. `feat: EBM head + isotonic calibration`
6. `feat: fairness metrics + counterfactual flip-rate harness`
7. `test: fairness gates in CI`
8. `feat: Kaggle benchmark script with honest caveats`
9. `chore: gitignore + remove committed artifacts`
10. `docs: collapse markdown forest, align README with reality`

Stop after each commit and run the test suite. Do not batch.

## Final note

If something in the existing code is unclear or seems contradictory, **ask before deleting**. The goal is an honest baseline, not a rewrite-from-scratch. Preserve mathematically sound code (e.g. `fairness_v2.py` bootstrap CIs, Beta posterior math) and rehouse it. Delete only what is wrong, redundant, or unshipped.

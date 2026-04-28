# Proposal Charter — Banking PoC Pitch

**READ THIS FIRST.** Every other prompt file in `handoffs/` assumes you've read this one.

## What we are building (one paragraph)

A demo + proposal package to win a Proof-of-Concept contract with a regulated banking client. The deliverable is **not** a production hiring system. The deliverable is **evidence that we can build one that survives the bank's Model Risk Management (MRM) review**. Every artifact (UI, PDF, slide, code comment) is judged by whether it would help or hurt that survival.

## The non-negotiable

> **This system reports AUC ~0.62 ± 0.06 on synthetic data. No deliverable — UI copy, slide, PDF, code comment, README — may imply higher performance, production readiness, or validation on real hiring outcomes. Banking compliance reviewers will check. The honesty of the numbers is the product.**

This sentence appears verbatim in every prompt file in `handoffs/`. If you find yourself wanting to soften it, stop and ask the user.

## Branch and ordering

- All work on branch `claude/code-review-zBiek`. Create a new branch only if user explicitly asks.
- Commit per work item. Do not batch.
- Push after each commit.

## Execution order across handoff files

```
PROPOSAL_CHARTER.md       (this — read first)
ANTIPATTERNS.md           (read second — what NOT to do)
PROMPT_01_ARCHITECTURE.md (must complete before 02 / 03)
PROMPT_02_DEMO.md         ┐
                          ├ can run in parallel after 01 lands
PROMPT_03_COLLATERAL.md   ┘
```

## Out of scope (do not work on these)

- Real banking data — we won't have it until the PoC contract is signed
- Production deployment, on-prem packaging, SOC 2 certification
- Real-time scoring, ATS integration code (Workday adapters etc.)
- LLM-based features of any kind (banks reject opaque LLMs in MRM submissions)
- Pushing AUC higher (we already established the synthetic data is the bottleneck — see `baselines.json`)
- Adding new fairness gates or metrics beyond what's in `tests/fairness/`
- Replacing or reorganizing the existing test suite

## Definition of done for this proposal cycle

After all three workstream prompts complete:

1. `streamlit run src/demo/app.py` produces a polished demo runnable end-to-end in ≤15 minutes
2. `handoffs/collateral/` contains: 12-slide deck markdown, 2-page exec summary, sample MRM model card PDF, sample fairness audit PDF, sample FCRA adverse action notice PDF, SOW template, CISO/TPRM one-pager
3. The architectural pivot from `PROMPT_01` is reflected in both demo and PDF outputs (rules + posteriors → per-skill aptitude scores with uncertainty intervals)
4. Test suite passes (current state: 159 passing, 1 expected failure for biased-fixture cascade — keep it that way)
5. `benchmark.json` numbers match what every artifact claims, byte-for-byte

## Honesty conventions

Every numeric claim in any artifact MUST trace to one of:
- `benchmark.json` (the canonical measured numbers)
- `baselines.json` (the n=10 seed sweep)
- A literal value in code that's covered by a test
- The marker `[ILLUSTRATIVE — sample data, not from this system]` displayed at the same visual prominence as the number

If you cannot trace a number, do not include it.

## Banker-language reminder

Bankers are not impressed by "explainable AI" or "audit-ready by construction" — they hear marketing. Use:
- "Decomposable to source rule firings with quantified posterior uncertainty"
- "Maps to specific SR 11-7 sections (purpose, theory, assumptions, limitations, monitoring)"
- "Complementary to existing ATS investments (does not replace Workday/Greenhouse)"
- "Independent auditor compatible (NYC LL144 self-certification is not claimed)"

Do not use:
- "Audit-ready by construction" (cut entirely)
- "Eliminates bias" (use "measures and bounds")
- "Powered by AI" / "Generative AI" (banker-trigger words)
- Vendor names in any deliverable except as competitive context in user-facing notes

## The single ask of every prompt

> Before starting work, re-read this charter. After every work item, before committing, ask yourself: "Would this artifact survive a banking MRM review?" If no, stop and fix.

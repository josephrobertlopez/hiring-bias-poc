# PROMPT_02_REMEDIATION_v5 — finish self-heal; fix zero-rules architecture bug

**Prerequisite:** read `handoffs/PROPOSAL_CHARTER.md` and `handoffs/ANTIPATTERNS.md` first. Continues from `PROMPT_02_REMEDIATION_v4` commits `7e3a261..26181ea`.

> **This system reports AUC ~0.62 ± 0.06 on synthetic data. No deliverable — UI copy, slide, PDF, code comment, README — may imply higher performance, production readiness, or validation on real hiring outcomes. Banking compliance reviewers will check. The honesty of the numbers is the product.**

## Evidence of remaining defects

**Defect 1** — Item 1 self-heal does not detect a poisoned single entry:
```
$ python -c "import json; open('audit_ledger.jsonl','w').write(json.dumps({'decision_id':'x','timestamp':'deterministic_ts_x','model_version':'1.0.0','full_scoring_payload':{},'fairness_metrics_at_decision_time':{}})+chr(10))"
$ PYTHONPATH=. python src/demo/smoke_test.py 2>&1 | tail -3
Smoke test results: 5 passed, 2 failed
```

**Defect 2** — `26181ea` commit body has unexpanded `$(cat /tmp/diagnosis_output.txt)` placeholder. The diagnostic output was never pasted; the diagnosis is unverified.

**Defect 3** — Real diagnostic script output (run independently against current `HEAD`):
```
Rules mined: 0
Posteriors fit: 8
AttributeError: 'dict' object has no attribute 'skill_tokens'
```

Zero rules mined on the demo dataset. The agent's prior "skill name mismatch" diagnosis was wrong — there are no rules to mismatch.

## Verification rule (carried from v4, repeated for emphasis)

**Acceptance commands must be run AFTER `git commit`, on the committed tree.** Paste the literal terminal output into the commit message body. Do not use shell heredoc substitution like `$(cat file)` — write the output literally with copy/paste from your terminal. The recurring failure mode on this branch is the agent pasting "passing" output that doesn't match post-commit reality.

## Scope ceiling

- Max LOC delta: ~150
- Max new commits: 3
- No new files (except an optional throwaway diagnostic script under `/tmp/`, never committed)

## Work items

### Item 1 — Fix the self-heal to actually detect poisoned entries

In `src/demo/components/data_loaders.py::populate_demo_ledger()`:

The current logic likely checks "does ledger file exist?" and stops there. Replace with: **read every line, attempt `datetime.fromisoformat(entry['timestamp'].rstrip('Z'))` on each, and if ANY entry raises `ValueError` (or KeyError on missing timestamp), truncate the file and rebuild from scratch.**

Pseudocode:
```python
def _ledger_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts = entry.get('timestamp', '')
                datetime.fromisoformat(ts.rstrip('Z'))  # raises if bad
        return True
    except (ValueError, KeyError, json.JSONDecodeError):
        return False

def populate_demo_ledger(...):
    if not _ledger_is_valid(LEDGER_PATH):
        if LEDGER_PATH.exists():
            print("audit_ledger.jsonl invalid; rebuilding", file=sys.stderr)
            LEDGER_PATH.unlink()
        # ... existing rebuild logic
```

Acceptance — paste literal output of BOTH commands in the commit message body:

```bash
# A: clean ledger
rm -f audit_ledger.jsonl
PYTHONPATH=. python src/demo/smoke_test.py 2>&1 | tail -3
# Must contain: "Smoke test results: 7 passed, 0 failed"
```

```bash
# B: poisoned single entry triggers rebuild
python -c "import json; open('audit_ledger.jsonl','w').write(json.dumps({'decision_id':'x','timestamp':'deterministic_ts_x','model_version':'1.0.0','full_scoring_payload':{},'fairness_metrics_at_decision_time':{}})+chr(10))"
PYTHONPATH=. python src/demo/smoke_test.py 2>&1 | tail -5
# Must contain: "Smoke test results: 7 passed, 0 failed"
```

If B fails, the self-heal is not fixing what it claims. Iterate before committing.

Commit message: `fix: self-heal detects poisoned individual ledger entries (not just missing file)`

### Item 2 — Diagnose why zero rules are mined (real diagnosis this time)

Run this script and paste the **literal output** into the commit message body. Copy/paste the terminal output yourself — do not use shell substitution:

```bash
PYTHONPATH=. python -c "
from src.demo.components.data_loaders import load_sample_data, get_demo_model_components

resumes, roles = load_sample_data()
extractor, miner, posteriors = get_demo_model_components()

print('Miner type:', type(miner).__name__)
print('Miner attributes:', [a for a in dir(miner) if not a.startswith('_')])
mined = getattr(miner, 'mined_rules', None) or getattr(miner, 'rules', None) or []
print('Rules mined:', len(mined))
print()

print('Sample resumes container type:', type(resumes).__name__)
first_resume = list(resumes.values())[0]
print('First resume type:', type(first_resume).__name__)
if isinstance(first_resume, dict):
    print('First resume keys:', list(first_resume.keys()))
else:
    print('First resume attrs:', [a for a in dir(first_resume) if not a.startswith('_')][:15])
print()

print('Posteriors type:', type(posteriors).__name__)
print('Posteriors length:', len(posteriors) if hasattr(posteriors, '__len__') else 'unknown')
if posteriors and hasattr(posteriors, 'keys'):
    print('First posterior key:', list(posteriors.keys())[0])
"
```

Then state the diagnosis as exactly one of:

- **(a)** Rule miner is configured correctly but `support`/`confidence`/`lift` thresholds reject all candidate rules on this small synthetic dataset → fix is to lower thresholds for the demo or expand sample data
- **(b)** Rule miner is never actually called in `get_demo_model_components()` → fix is to add the call
- **(c)** Rule miner is called but posteriors are fit on something *other* than rules (explains the "8 posteriors on 0 rules" anomaly) → fix is to align the inputs
- **(d)** Type mismatch: rule miner / scorer expects `Resume` objects but receives dicts → fix is the converter at the boundary

Include 1–2 sentences of evidence from the diagnostic output supporting the choice.

**Do not implement a fix in this commit.** Diagnosis only. The architectural correction belongs in Item 3 with full scope awareness.

Commit message: `diagnose: zero-rules root cause (real, with literal script output)`

### Item 3 — Implement the fix dictated by the Item 2 diagnosis

Based on Item 2's diagnosis letter (a/b/c/d), implement the minimal fix:

- **(a)**: lower `RuleMinerConfig` thresholds for the demo path only (`support=0.05`, `confidence=0.4`, `lift=1.0`); document in a code comment that production thresholds will be tuned on real data
- **(b)**: wire the rule miner call into `get_demo_model_components()` between extractor and posterior fitting
- **(c)**: ensure `fit_rule_posteriors` receives the actual mined rules; resolve the source of the "8 posteriors" mystery (likely posteriors are over candidates or features, not rules)
- **(d)**: convert dicts to `Resume` objects at the boundary in `load_sample_data()` or `get_demo_model_components()`

After the fix, run all three of these commands and paste the literal output into the commit message body:

```bash
# 1: confirm rules are now mined
PYTHONPATH=. python -c "
from src.demo.components.data_loaders import get_demo_model_components
extractor, miner, posteriors = get_demo_model_components()
mined = getattr(miner, 'mined_rules', None) or getattr(miner, 'rules', None) or []
print(f'Rules mined: {len(mined)}')
"
# Must produce: Rules mined: N where N >= 5
```

```bash
# 2: smoke test passes from clean state
rm -f audit_ledger.jsonl
PYTHONPATH=. python src/demo/smoke_test.py 2>&1 | tail -3
# Must produce: Smoke test results: 7 passed, 0 failed
```

```bash
# 3: audit ledger contains real firing rules
PYTHONPATH=. python -c "
import json
total_rules_fired = 0
n_decisions = 0
with open('audit_ledger.jsonl') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        n_decisions += 1
        for skill, apt in d['full_scoring_payload']['aptitudes'].items():
            total_rules_fired += len(apt.get('contributing_rules', []))
print(f'Decisions: {n_decisions}, total firing rules across all decisions: {total_rules_fired}')
"
# Must produce: total firing rules > 0
# (If still 0, the entire 'Bayesian rule posteriors' pitch in PROMPT_03's PDFs would be empty)
```

Commit message: `fix: <one-line summary based on diagnosis letter>` — for example, `fix: lower demo rule miner thresholds to produce non-empty mined_rules`

## Definition of done

- 3 commits pushed to `origin/claude/code-review-zBiek`
- Each commit body contains literal acceptance command outputs (no `$(cat ...)` placeholders, no paraphrasing)
- After Item 3: at least one decision in `audit_ledger.jsonl` has non-empty `contributing_rules` for at least one skill
- After Item 3: smoke test passes from both clean and poisoned ledger states

## Stop conditions

- Item 1 acceptance B fails → stop and re-investigate the self-heal logic; do not commit until it passes
- Item 2 diagnostic script crashes → stop, paste the traceback into the commit, that traceback IS the diagnosis
- Item 3 still produces `Rules mined: 0` → stop and ask before committing; the chosen fix didn't work and we need to reconsider
- LOC delta exceeds 150 → stop and ask
- You catch yourself pasting `$(cat ...)` or any other unexpanded shell substitution into a commit message → stop, copy/paste the literal terminal output instead

## What NOT to do

- Do not start `PROMPT_03_COLLATERAL.md` until all 3 items land cleanly
- Do not declare a diagnosis without running the diagnostic script and pasting its real terminal output
- Do not lower the fairness filter thresholds in the rule miner — those are correctness gates, not tuning knobs. The `RuleMinerConfig.fairness_filter_*` settings stay as-is. Only the support/confidence/lift thresholds may be tuned for the demo, and only with a comment explaining why
- Do not synthesize fake rules to make the demo look better
- Do not edit any file outside `src/demo/components/`, `src/features/rule_miner.py`, `src/posteriors/rule_reliability.py`, `src/aptitude/scorer.py`, or test files
- Do not paraphrase command output in commit messages — paste literal terminal output verbatim
- Do not commit `audit_ledger.jsonl` even though it gets populated during testing (it's gitignored — verify with `git ls-files | grep audit_ledger` returning empty before pushing)

## After all 3 items land

Run a final sanity check and post the output in a comment on the branch:

```bash
PYTHONPATH=. pytest tests/ -p no:cacheprovider --no-cov -o addopts="" --tb=no -q 2>&1 | tail -3
python src/demo/smoke_test.py 2>&1 | tail -3
grep -rnE "time\.sleep|np\.random\.uniform|mock_decisions|mock_delta|create_mock_audit_decisions|deterministic_ts" src/
```

Expected: pytest reports `177+ passed, 1 failed` (only `test_overall_fairness_pipeline_gate` cascade); smoke test `7 passed, 0 failed`; grep returns empty.

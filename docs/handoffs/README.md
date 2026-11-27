# Codex (GPT-6.2 Astra) Handoff Protocol

Per [`AWST_Research_Protocol_v0.3.md`](../../AWST_Research_Protocol_v0.3.md) Section 14.1, Codex with **GPT-6.2 Astra** is invoked strictly through **manual researcher-mediated handoffs**. Antigravity must never attempt to automate the researcher's Codex account, credentials, quota, or UI.

## Current Status

| Handoff ID | Scope | Target | Status |
|---|---|---|---|
| [`H001_rq6_competency_pilot.md`](H001_rq6_competency_pilot.md) | RQ6 synthetic toy-network competency pilot (Tasks A, B, C) | GPT-6.2 Astra (Codex) | **COMPLETED & VERIFIED** (`3/3`, `\$1.10` in-turn shadow / `\$0.00` direct API; [`results/m2_rq6_competency_comparison.json`](../../results/m2_rq6_competency_comparison.json)) |
| [`H002_stage1_awst_t1.md`](H002_stage1_awst_t1.md) | Stage 1 `C1` (donor-free) + `T1` (weights-only AWST, `2B -> 0.8B`) | GPT-6.2 Astra (Codex) | **COMPLETED & VERIFIED** (`T1` negative on validation: `57.0%` GSM8K vs `62.0%` `C0_raw`; [`results/m2_h002_verification.json`](../../results/m2_h002_verification.json)) |
| [`H003_stage1_awst_t2.md`](H003_stage1_awst_t2.md) | Stage 1 `T2` (activation-assisted AWST, `2B -> 0.8B`) | GPT-6.2 Astra (Codex) | **COMPLETED & VERIFIED** (`T2` negative on validation: `57.0%` GSM8K vs `62.0%` `C0_raw`; [`results/m2_h003_verification.json`](../../results/m2_h003_verification.json)) |

## Required Handoff Procedure

1. Create `docs/handoffs/HNNN_task.md` (zero-padded: `H001_...`, `H002_...`).
2. Update `docs/progress.md` before pausing so the repository reflects the pending handoff.
3. Ask the researcher to submit the self-contained prompt to GPT-6.2 Astra in Codex and pause until they return the response or patch.
4. Upon return:
   - Log exact model version, prompt file, response summary, tool calls, token/dollar cost (or shadow API cost), and attribution in `docs/progress.md` and the corresponding run manifest.
   - Independently review and test every proposed script, edit, or numerical claim before accepting it into the codebase or results.
   - Keep external experimental runs isolated on clean branches or manifests so Gemini's analysis and Astra's edits never cross-contaminate matched comparisons.

## Handoff File Template (`HNNN_task.md`)

```markdown
# HNNN: <Short Task Title>

- **Date:** YYYY-MM-DD
- **Target Agent / Harness:** GPT-6.2 Astra via Codex
- **Authorization Mode:** `REVIEW-ONLY` | `AUTHORIZED IMPLEMENTATION (requires M0/ToS/pilot approval)`
- **Repository Commit SHA:** `<exact sha>`
- **Model / Checkpoint Revisions:** `<exact HF repo + pinned commit SHA, or N/A>`
- **Budget / Resource Limits:** Max tokens, max tool calls, max forward passes, max edit count, dollar cap
- **Relevant Files (Read-Only / Editable):** Explicit list

## 1. Purpose and Objective
Clear statement of the task and what "done" looks like.

## 2. Background and Constraints
- Minimum necessary context (NO hidden test questions/answers, NO secrets/tokens, NO full raw weight dumps).
- Architectural invariants, training-free rules, and track separation (`T1` weights-only vs `T2` activation-assisted).

## 3. Available Tools and Allowed Operations
Exact scripts, Python APIs, and development data splits permitted.

## 4. Acceptance Tests and Expected Deliverables
Exact output format, provenance log fields, and verification commands Antigravity will run on return.
```

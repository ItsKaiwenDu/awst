# AGENTS.md — Shared Operating Rules for Project AWST

This file defines the shared operating rules for all AI agents working on Project AWST across **Google Antigravity** and **Codex**. The canonical specification is [`AWST_Research_Protocol_v0.3.md`](AWST_Research_Protocol_v0.3.md); if any summary here conflicts with the protocol or explicit researcher approvals, the protocol and researcher decisions govern.

---

## 1. Harness Roles and Handoff Discipline

| Harness | Model | Role |
|---|---|---|
| **Google Antigravity** | Gemini 4 Neon / Carbon (as approved) | **Primary research coordinator:** overall coordination, infrastructure, experimental design, baseline/control implementation, independent verification, documentation, Git workflow, and paper writing. |
| **Codex** | GPT-6.2 Astra | **Secondary specialist & primary weight-space programmer:** designated weight-space programmer for core AWST weight transplantation experiments, subject to ToS clearance and human approval. |

- **Manual handoff rule:** Antigravity must **never** attempt to access Codex directly or automate the researcher's Codex account, credentials, quota, or UI. Every Codex task must be written as a self-contained handoff brief in `docs/handoffs/HNNN_task.md` (following [`docs/handoffs/README.md`](docs/handoffs/README.md)) and paused for the researcher to submit and return.
- **Cross-harness isolation & verification:** Review, independently verify, and test any patch or recommendation returned from Codex before integrating. For matched agent comparisons (RQ6), keep prompts, tool budgets, and workspaces strictly isolated so neither agent's analysis contaminates the other's independent trial.

---

## 2. Scientific Integrity and Experimental Constraints

1. **Never fabricate or conceal:** Never invent numbers, citations, file paths, or test results. Negative results, replication failures, and blocked runs are valid scientific outcomes and must be logged transparently in `docs/progress.md` and run manifests.
2. **Training-free & fixed-architecture invariants:** Primary AWST (`T1`, `T2`) forbids gradient descent, backpropagation, loss-driven parameter fitting, fine-tuning, distillation, or learned adapters. Every modified checkpoint must verify **bit-exact identical** parameter count, tensor shapes, layer count, tokenizer, and runtime graph, plus unchanged frozen components (e.g., vision encoder).
3. **Track separation:** Keep weights-only (`T1`, no donor forward passes) strictly separated from activation-assisted (`T2`, donor activations permitted). Never pool `T1` and `T2` results or silently switch tracks.
4. **Hidden test isolation:** Final held-out test sets and answers must remain **outside** the agent-readable repository (see `docs/decisions.md` `D006`). Never expose hidden test questions, answers, or benchmark-derived optimization signals to the editing agent.
5. **Pristine weights & immutable provenance:** Never overwrite original donor or recipient checkpoints. Log every edit with exact donor tensor provenance, transformation formula, target recipient slot, changed parameter count, and relative Frobenius norm.
6. **Bounded inspection:** Never serialize entire weight matrices into an agent's context window. Use local numerical tools for statistics and bounded slices; enforce strict byte, forward-pass, token, and trial budgets.
7. **Literature & citation hygiene:** Never commit copyrighted PDFs (`docs/literature/*.pdf` is gitignored). Verify bibliographic details from primary sources before citing.

---

## 3. Approval Gates (Pause Before Proceeding)

Stop and request explicit researcher approval before:
- Any provider-hosted model-weight improvement work while OpenAI/Google competing-model ToS compliance remains unresolved (`docs/permissions.md`).
- Any paid API call, cloud GPU job, or budget reallocation (`$20` pilot gate; `$200` hard ceiling).
- Any model checkpoint download (verify license, pinned revision, and 30 GB Stage 0–2 cache cap first; **never** auto-download `Kimi-K3.1-Pro` or exceed the separate 1 GB Kimi remote pilot cap without Kimi Escalation Gate approval).
- Any weight-modification experiment or external Codex handoff (`docs/handoffs/HNNN_task.md`).
- Any destructive operation, protocol change after inspecting validation/test results, or repository visibility / remote / history changes.

---

## 4. Session Start and Context Recovery

Do **not** rely on automatic instruction loading. At the start of every session or after context compaction:
1. Read `AGENTS.md`, `GEMINI.md` (if in Antigravity), `AWST_Research_Protocol_v0.3.md` (or check relevant sections), and `docs/progress.md`.
2. Read `docs/decisions.md`, `docs/permissions.md`, and any pending `docs/handoffs/HNNN_task.md`.
3. Run `git status` and `git log -n 5 --oneline` to reconcile repository state.
4. Resume only from the first unfinished **approved** task.

---

## 5. Persistent Logging and Git Workflow

- Record every meaningful work unit (date, milestone, task IDs, changed paths, commit SHA, test outputs, budget spend, blockers, next action) in `docs/progress.md` before pausing, handing off, or ending a session.
- After each major completed step:
  1. Update `docs/progress.md`, checklists, and `docs/decisions.md`.
  2. Run verification checks and inspect `git status` and `git diff`.
  3. Stage only reviewed, safe files (verify zero secrets, model binaries, hidden eval data, copyrighted PDFs, or large raw logs).
  4. Commit with a concise title (`<70` characters) and structured numbered body summarizing what changed and why.
  5. Push to the approved remote (`origin`) and branch (`main`) without force-pushing or rewriting history, and verify push success before moving on.

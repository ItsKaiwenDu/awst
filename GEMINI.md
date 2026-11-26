# GEMINI.md — Antigravity Entry Point for Project AWST

This file is the Antigravity/Gemini entry point. Shared rules live in [`AGENTS.md`](AGENTS.md) and the canonical protocol lives in [`AWST_Research_Protocol_v0.3.md`](AWST_Research_Protocol_v0.3.md).

## Mandatory Session-Start Checklist

Always explicitly read these files at session start or after compaction (do not assume automatic loading):
1. [`AGENTS.md`](AGENTS.md)
2. [`AWST_Research_Protocol_v0.3.md`](AWST_Research_Protocol_v0.3.md)
3. [`docs/progress.md`](docs/progress.md)
4. [`docs/decisions.md`](docs/decisions.md) and [`docs/permissions.md`](docs/permissions.md)
5. Any pending brief in `docs/handoffs/`

## Antigravity-Specific Notes

- **Role:** Primary research coordinator (infrastructure, experimental design, baselines/controls, independent verification, documentation, Git workflow, and paper writing).
- **Codex handoffs:** GPT-6.2 Astra (Codex) is the designated primary weight-space programmer for core AWST transplantation experiments. Prepare `docs/handoffs/HNNN_task.md` and pause for the researcher to relay it; never automate Codex.
- **Git commits:** Format commit messages using a concise Title Case title with the `Commit: ` prefix (`<70` chars, e.g., `Commit: Add M1 Evaluation Harness`) and a numbered list body (matching the workspace `git-commit` skill), then push to `origin/main` after verification. Do not make an extra commit solely to record the previous commit's SHA.
- **Narration:** Provide brief, clear progress updates alongside tool calls so the researcher has continuous visibility into active steps.

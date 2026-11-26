# Agentic Weight-Space Transplantation (AWST)

**Working paper title:** *Agentic Weight-Space Transplantation: Training-Free Transfer of Mathematical Reasoning Between Language Models*  
**Status:** Milestone M2b in progress (`H001` RQ6 pilot verified `3/3` for both Neon & Astra; awaiting Codex `H002`–`H003`)  
**Primary domain:** Grade-school mathematical reasoning (narrow synthetic arithmetic diagnostic + GSM8K confirmatory benchmark)  
**Spend to date:** \$0.00 / \$200.00 hard ceiling (\$20.00 M2 pilot gate authorized)

> **Research Disclaimer (Academic Use Only):** Project AWST is an open, non-commercial academic feasibility study on training-free weight transfer between open-weight Apache-2.0 models (`Qwen3.5`). It is **not** intended for distillation or for developing models that compete with Google, OpenAI, or any other provider. GPT-6.2 Astra acts strictly as a weight-space transplanter between open-weight checkpoints.

---

## Overview

AWST investigates whether an AI agent can selectively identify, extract, adapt, and transplant learned mathematical reasoning weights from a larger open-weight donor model into a smaller recipient model **without gradient-based training** and while **preserving the recipient's exact architecture, tensor shapes, and parameter count**.

Negative and mixed outcomes are treated as valid scientific results provided the baselines, controls (`C0`–`C5`), and provenance checks are executed rigorously.

## Repository Layout

```text
awst/
  AWST_Research_Protocol_v0.3.md  # Canonical research protocol
  AGENTS.md                       # Shared operating rules for Antigravity and Codex
  GEMINI.md                       # Antigravity / Gemini session entry point
  README.md                       # Project overview and quick links
  LICENSE                         # MIT License
  .gitignore                      # Excludes binaries, secrets, hidden tests, PDFs, caches
  docs/
    progress.md                   # Live milestone checklist, budget, blockers, next actions
    decisions.md                  # Numbered research & engineering decisions (D001–D008)
    permissions.md                # Model licenses, provider ToS, pricing, compute & storage audit
    preregistration.md            # Provisional experimental & statistical preregistration (draft)
    research_protocol.md          # Pointer + launch-time harness role addendum
    literature_review.md          # Detailed review & novelty reassessment (9 core/recent papers)
    literature/
      README.md                   # Literature tracking table and PDF policy
      bibliography.bib            # Verified BibTeX citations
    handoffs/
      README.md                   # Manual Codex (GPT-6.2 Astra) handoff protocol & template
  configs/                        # Pinned model and experiment configs (M1+)
  src/                            # Research harness packages (M1+)
  tests/                          # Automated unit & integrity tests (M1+)
  data/                           # Development and validation splits ONLY (hidden test lives outside repo)
  runs/                           # Immutable run manifests and logs (large raw logs gitignored)
  results/                        # Machine-readable evaluation tables
  paper/                          # Manuscript drafts (M4)
```

> [!IMPORTANT]
> **Data isolation:** Frozen held-out test sets and labels are stored strictly **outside** this repository (see `docs/decisions.md` `D006`) and must never be accessed during development or agent-directed editing. No model checkpoints or empirical results exist at Milestone M0.

## Verified Local Environment (M0 Audit)

- **Machine:** Apple M4 Max (14 CPU cores, 36 GiB / 38,654,705,664 bytes unified memory, macOS 27.2.1)
- **Storage:** 926 GiB total, **580 GiB available** on `/` (Stage 0–2 local cache cap: 30 GB)
- **Installed Python binaries:** Python 3.14.5 (`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`) and Python 3.11 (`/opt/homebrew/bin/python3.11`). ML dependencies (`torch`, `transformers`, `safetensors`, `datasets`, `pytest`) are not yet installed (deferred to M1 `.venv` setup).

## Key Documents

- [AWST_Research_Protocol_v0.3.md](AWST_Research_Protocol_v0.3.md) — Canonical research specification
- [AGENTS.md](AGENTS.md) — Shared rules for Antigravity and Codex
- [GEMINI.md](GEMINI.md) — Antigravity / Gemini entry point
- [docs/progress.md](docs/progress.md) — Current milestone state, checklist, and pending human approvals
- [docs/literature_review.md](docs/literature_review.md) — Verified literature review and novelty analysis
- [docs/permissions.md](docs/permissions.md) — License, ToS, pricing, and hardware audit
- [docs/decisions.md](docs/decisions.md) — Open and settled research/engineering decisions
- [docs/preregistration.md](docs/preregistration.md) — Provisional preregistration (not yet frozen)

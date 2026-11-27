# Agentic Weight-Space Transplantation (AWST)

**Title:** *Agentic Weight-Space Transplantation: An Empirical Feasibility Study on Training-Free Mathematical Transfer Between Language Models*  
**Status:** **Milestones M0–M4 Complete** (awaiting human approval prior to any public release or publication)  
**Primary Domain:** Grade-school mathematical reasoning (multi-digit carry/borrow arithmetic + GSM8K)  
**Spend to Date:** **\$0.00** direct API/cloud spend / \$200.00 hard ceiling (**\$14.2912** cumulative Codex shadow across `H001`–`H004` inside the authorized \$20.00 M2 pilot gate)

> **Research Disclaimer (Academic Use Only):** Project AWST is a non-commercial academic feasibility study on training-free weight transfer between open-weight Apache-2.0 models (`Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B`). It does **not** perform distillation or develop models that compete with any provider; agent programmers act strictly as transparent weight-space editors between pinned open-weight checkpoints.

---

## Overview & Key Empirical Findings

AWST investigates whether an AI agent can selectively identify, extract, adapt, and transplant learned mathematical reasoning weights from a larger donor model into a smaller fixed-architecture recipient **without gradient-based training** (`0` backpropagation steps, unchanged `873,438,784` stored / `852,985,920` active runtime parameters, and `153/153` frozen multimodal vision tensors). We strictly separate **weights-only (`T1`)** editing from **activation-assisted (`T2`)** editing and evaluate against unmodified baselines (`C0_raw`, `C0_loopfixed`), donor-free WSP (`C1`), random perturbations (`C2`), unaligned donor copying (`C3`), conventional algorithmic transfer (`C4_weights` SVD/Hungarian; `C4_act` APM+ridge), and shuffled-donor controls (`C5`).

Under a frozen greedy `enable_thinking=False`, `1,024`-token protocol (`nothink 1024`) on Apple M4 Max MPS (`bfloat16`) that avoids `0.8B` reasoning-loop confounds while retaining a `+21.5 pp` isolated GSM8K and `+44.0 pp` carry-arithmetic donor lead (`Qwen3.5-2B` vs. `C0_raw`):

1. **Neither frozen Stage 1 checkpoint met preregistered joint success criteria:** On isolated confirmatory evaluation (`n=200` GSM8K, `n=150` synthetic arithmetic, `n=100` OOD), weights-only `T1` (`H002`, 9,050 digit-embedding entries) scored `54.5%` GSM8K (`+3.0 pp [-2.0, +8.0]` vs `C0_raw` `51.5%`, `p = 0.345`), failing to exceed `C1` (`+1.5 pp`, Holm `p = 0.701`), `C4_weights` (`+7.5 pp`, Holm `p = 0.080`), or shuffled-donor `C5_T1_rowperm` (`+0.5 pp`, `p = 1.000`). Activation-assisted `T2` (`H003`, 2 DeltaNet Layer 13 Head 12 scalars) scored `51.5%` GSM8K (`0.0 pp [-3.5, +3.5]` vs `C0_raw`), `-3.5 pp [-8.5, +1.0]` vs `C5_T2_norm_matched`, and `-5.0 pp [-10.0, -1.0]` (`p = 0.0625`) on OOD templates. Because `T1`'s GSM8K interval spans `+5 pp` and each track reflects a single editing session (`N=1`), the result establishes failure on the joint criteria rather than proving zero effect across arbitrary multi-session or intermediate-scale strategies.
2. **Selective OOD Fragility of Unaligned Copying (`H3`):** Random unaligned donor MLP slice copying (`C3`) suffered a `-20.0 pp [-29.0, -11.0]` OOD collapse (`p = 3.59e-5`) and degraded 10-passage prose perplexity (`62.63 -> 78.87`), which aligned `C4` and ultra-sparse edits avoided.
3. **Selective Remote Safetensors Retrieval (`H4`):** HTTP `206` Byte-Range requests fetched exact target tensors from Hugging Face using `176,584` of `1,746,942,600` shard bytes (`9,892.98x` reduction, verified bit-exact).
4. **Stage Gate (`D011`):** Following negative Stage 1 validation outcomes, Stages 2 (`Qwen3.5-9B`), 3 (`Gemma 4 E2B`), and 4 (`Kimi K3.1 Pro`) were formally skipped prior to isolated evaluation.

---

## Primary Deliverables

- **Manuscript:** [`paper/awst_paper.md`](paper/awst_paper.md) (complete Sections 1–9 + Appendices, aligned with `H004` independent audit)
- **Publication Figures:** [`paper/figures/`](paper/figures/) (`fig1_split_trajectory`, `fig2_m3_forest_contrasts`, `fig3_edit_magnitude_vs_ood_ppl` in `.png` and `.pdf`)
- **Final Artifact Manifest & Expense Ledger:** [`paper/artifact_manifest.md`](paper/artifact_manifest.md)
- **Canonical Protocol & Records:** [`AWST_Research_Protocol_v0.3.md`](AWST_Research_Protocol_v0.3.md), [`docs/preregistration.md`](docs/preregistration.md), [`docs/decisions.md`](docs/decisions.md) (`D001`–`D011`), [`docs/progress.md`](docs/progress.md), [`docs/permissions.md`](docs/permissions.md)
- **Codex Handoffs & Independent Verification:** [`docs/handoffs/`](docs/handoffs/) (`H001`–`H004`), [`runs/codex_h004/review.md`](runs/codex_h004/review.md), [`results/m3_isolated_evaluation.json`](results/m3_isolated_evaluation.json)

---

## Repository Layout

```text
awst/
  AWST_Research_Protocol_v0.3.md  # Canonical research protocol
  AGENTS.md / GEMINI.md           # Shared and Antigravity harness instructions
  README.md                       # Project overview, findings & reproduction guide
  requirements.txt                # Pinned Python 3.11.16 dependencies
  docs/                           # Protocol addenda, D001-D011 decisions, progress log, literature, handoffs
  configs/                        # Pinned model revisions and environment metadata
  src/
    data/                         # Deterministic arithmetic & GSM8K split generators
    weight_io/                    # HTTP 206 Range reader & Safetensors invariant utilities
    interventions/                # Immutable patch applier & Stage 0 loopfix replay
    evaluation/                   # Runner, parser, retention/perplexity suite, M2/M3 verifiers
    baselines/                    # C2, C3, C4_weights, C4_act, C5 implementations
    agent_interface/              # Bounded AgentWeightToolbox & RQ6 toy competency suite
    paper/                        # Publication figure generator & automated claim/citation auditor
  tests/                          # Automated unit & paper-verification tests (pytest)
  data/                           # Development & validation splits ONLY (hidden test lives outside repo)
  runs/                           # Immutable run manifests and Codex H001-H004 artifacts
  results/                        # Machine-readable M1-M3 empirical JSON tables
  paper/                          # Final manuscript (awst_paper.md), BibTeX, figures, artifact_manifest.md
```

> [!IMPORTANT]
> **Data Isolation (`D006`):** Held-out isolated test sets (`gsm8k_test`, `synthetic_test`, `synthetic_ood_test`) and M3 per-item logs live strictly outside this repository at `/Users/kaiwendu/.awst_isolated_eval/`. Freeze commit `d12e8fb` preceded all isolated test reads (`42b859b`).

---

## Environment & Reproduction

- **Hardware:** Apple M4 Max (36 GiB unified memory, 14 CPU cores, macOS `27.2.1`)
- **Python Environment:** Pinned `.venv` (`Python 3.11.16`, `torch 2.15.0` MPS `bfloat16`, `transformers 5.24.1`, `safetensors 0.8.0`, `datasets 5.2.0`, `scipy`, `pytest 9.2.0`); figures rendered via Python `3.14.5` (`matplotlib 3.11.1`).

```sh
# Verify all unit tests and paper/manifest/citation checks
.venv/bin/pytest -q

# Run the standalone paper claim, figure, and citation auditor
.venv/bin/python src/paper/verify_paper_numbers.py

# Regenerate figures from stored results/*.json
MPLCONFIGDIR=/tmp/mplconfig /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/paper/generate_figures.py
```

---

## Limitations & Release Gate

- Single editing session (`N=1`) per track precludes statistical estimation of run-to-run agent variability; GSM8K subset (`n=200`) provides moderate power for `+5 pp` effects (~`26%–40%` exact McNemar depending on discordance).
- `synthetic_arithmetic.py` generator and OOD templates are agent-readable in-repo; OOD tests unseen two-step wording rather than new carry algorithms; GSM8K parser uses a last-number fallback without a blinded human completion audit.
- `ret_ifeval` (`n=30`, `21/30` exact match) and single post-eval MPS memory snapshots (`2395.8 MB`) do not certify broad general-capability preservation or mid-generation peak memory bounds.
- Repository remains **private**; public release, publication, or weight distribution requires explicit human approval (`AWST_Research_Protocol_v0.3.md` Section 14.2 M4).

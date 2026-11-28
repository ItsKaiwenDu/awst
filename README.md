# Agentic Weight-Space Transplantation (AWST)

**Paper:** *Agentic Weight-Space Transplantation on Language Models*  
**Author:** Kaiwen Du (College of Arts and Sciences, Santa Clara University)  
**Manuscript:** [`paper/main.tex`](paper/main.tex) · [`paper/awst_paper.md`](paper/awst_paper.md) · [`paper/figures/`](paper/figures/)

This repository contains the code, experimental manifests, verification tools, and manuscript for our preregistered empirical feasibility study of **Agentic Weight-Space Transplantation (AWST)**. We test whether an AI agent can inspect model weights, adapt across hidden-dimension mismatches, and transplant grade-school math reasoning from a larger donor (`Qwen/Qwen3.5-2B`) into a smaller fixed-architecture recipient (`Qwen/Qwen3.5-0.8B`) with **zero gradient updates**. We separate a **weights-only** track (`T1`, zero donor forward passes) from an **activation-assisted** track (`T2`, bounded donor probing) and compare both against donor-free, random, conventional algorithmic, and shuffled-donor controls.

---

## Key Findings

Under a frozen greedy, no-thinking, 1,024-token evaluation protocol (`enable_thinking=False`) that eliminates `0.8B` reasoning-loop confounds while preserving a large donor lead (`+21.5 pp` on GSM8K, `+44.0 pp` on multi-digit carry arithmetic):

| Arm | GSM8K (`n=200`) | Carry (`n=50`) | OOD (`n=100`) | Prose PPL | Changed Params |
|---|---:|---:|---:|---:|---:|
| `C0_raw` (unedited `0.8B`) | `51.5%` | `30.0%` | `79.0%` | `62.63` | `0` |
| `Qwen3.5-2B` (donor) | **`73.0%`** | **`74.0%`** | **`81.0%`** | `41.02` | Reference donor |
| `C1` (donor-free WSP) | `53.0%` | `26.0%` | `78.0%` | `62.63` | `9,097` |
| `T1` (weights-only AWST) | `54.5%` | `28.0%` | `77.0%` | `62.62` | `9,050` |
| `T2` (activation-assisted AWST) | `51.5%` | `30.0%` | `74.0%` | `62.47` | `2` |
| `C4_weights` (SVD/Hungarian) | `47.0%` | `18.0%` | `72.0%` | `73.86` | `172.51M` |
| `C4_act` (APM + ridge) | `47.5%` | `26.0%` | `79.0%` | `66.50` | `168.33M` |
| `C3` (random unaligned slice) | `49.5%` | `20.0%` | **`59.0%`** | **`78.87`** | `173.20M` |

- **Negative single-session math transfer:** Neither frozen single-session edit (`T1` editing 10 digit embeddings; `T2` editing 2 Gated DeltaNet decay scalars) cleared our preregistered feasibility criteria or outperformed donor-free (`C1`) and shuffled-donor (`C5`) controls (`C5_T1_rowperm`: `54.0%`, `C5_T2_norm_matched`: `55.0%` on GSM8K).
- **Unaligned copying fragility vs. sparse/aligned edits:** Unaligned donor copying (`C3`) caused a `-20.0 pp` out-of-distribution collapse (`p = 3.59e-5`) and degraded prose perplexity (`62.63 -> 78.87`), which aligned and sparse edits avoided.
- **Bit-exact selective remote weight retrieval:** An HTTP `206` Byte-Range Safetensors reader reproduced target tensor slices bit-for-bit using `176,584` of `1,746,942,600` shard bytes (`9,892.98x` reduction).

---

## Quick Start

Create a virtual environment and install the pinned dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite, manuscript auditor, figure generator, or Overleaf packager via `make`:

```sh
make help       # List available targets
make test       # Run pytest unit and manuscript verification tests
make verify     # Run the standalone paper claim, LaTeX, figure, and citation auditor
make check      # Run both test and verify
make figures    # Regenerate publication figures from results/*.json
make overleaf   # Rebuild awst_overleaf.zip from paper/
```

---

## Repository Layout

```text
awst/
  Makefile                        # Convenience targets for tests, verification, figures, Overleaf zip
  requirements.txt                # Pinned Python dependencies
  AWST_Research_Protocol_v0.3.md  # Canonical research protocol
  configs/                        # Pinned model revisions and environment metadata
  data/                           # Development and validation splits only
  docs/                           # Preregistration, decision log, progress log, literature, handoffs
  paper/                          # LaTeX/Markdown manuscripts, BibTeX, figures, artifact manifest
  results/                        # Machine-readable evaluation JSON tables
  runs/                           # Immutable patch manifests and independent audit reports
  src/
    agent_interface/              # Bounded AgentWeightToolbox and synthetic competency suite
    baselines/                    # C2, C3, C4_weights, C4_act, and C5 implementations
    data/                         # Deterministic arithmetic and GSM8K split generators
    evaluation/                   # Evaluation runner, parser, retention/perplexity suite, verifiers
    interventions/                # Immutable Safetensors patch applier and loop-repair replay
    paper/                        # Figure generator and automated manuscript auditor
    weight_io/                    # HTTP 206 Range reader and Safetensors invariant utilities
  tests/                          # Automated unit and manuscript verification tests
```

Held-out confirmatory test splits (`gsm8k_test`, `synthetic_test`, `synthetic_ood_test`) and per-item evaluation logs were stored outside the repository (`<external_eval_dir>/`) during all editing sessions to preserve strict test isolation.

---

## Citation & Scope

```bibtex
@misc{du2026awst,
  title={Agentic Weight-Space Transplantation on Language Models},
  author={Du, Kaiwen},
  year={2026},
  note={College of Arts and Sciences, Santa Clara University},
  url={https://github.com/ItsKaiwenDu/awst}
}
```

This is a non-commercial academic feasibility study operating exclusively on open-weight Apache-2.0 checkpoints (`Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B`).

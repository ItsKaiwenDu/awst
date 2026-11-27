# Project AWST — Final Artifact Manifest & Expense Summary (Milestone M4)

**Date:** 2026-11-27  
**Repository:** `/Users/kaiwendu/Documents/GitHub/awst` (`origin/main`)  
**Freeze Reference Commit:** `d12e8fb` (M3 evaluation `42b859b`, `H004` review `9c975e9`)  
**Release Status:** Private research repository; **pending human approval before any publication, checkpoint distribution, or public release**.

---

## 1. Deliverable Artifact Index

| Category | Path | Role / Verification |
|---|---|---|
| **Manuscript & BibTeX** | [`paper/awst_paper.md`](awst_paper.md) | Complete empirical feasibility report following Protocol Section 13 & `H004` review |
| | [`paper/references.bib`](references.bib) | 15 verified primary references (byte-identical to `docs/literature/bibliography.bib`) |
| **Publication Figures** | [`paper/figures/fig1_split_trajectory.png`](figures/fig1_split_trajectory.png) (`.pdf`) | Blinded validation vs. isolated test accuracy (`C0`–`C5`, `T1`, `T2`, `Qwen3.5-2B`) |
| | [`paper/figures/fig2_m3_forest_contrasts.png`](figures/fig2_m3_forest_contrasts.png) (`.pdf`) | Paired differences, 95% bootstrap CIs, exact McNemar `p`, track Holm `p` |
| | [`paper/figures/fig3_edit_magnitude_vs_ood_ppl.png`](figures/fig3_edit_magnitude_vs_ood_ppl.png) (`.pdf`) | Whole-model relative Frobenius norm vs. OOD accuracy and prose perplexity (`H3`) |
| **Paper Tooling & Tests** | [`src/paper/generate_figures.py`](../src/paper/generate_figures.py) | Deterministic figure generation directly from `results/*.json` & `runs/` JSONs |
| | [`src/paper/verify_paper_numbers.py`](../src/paper/verify_paper_numbers.py) | Programmatic claim, citation, figure, and hidden-data hygiene auditor |
| | [`tests/test_paper_verification.py`](../tests/test_paper_verification.py) | Automated `pytest` enforcement of paper/manifest numeric & citation integrity |
| **Protocol & Audit Docs** | [`AWST_Research_Protocol_v0.3.md`](../AWST_Research_Protocol_v0.3.md) | Canonical project protocol |
| | [`docs/preregistration.md`](../docs/preregistration.md) | Frozen Pilot Protocol v1.0 & M3 confirmatory freeze addendum (§8) |
| | [`docs/decisions.md`](../docs/decisions.md) | Numbered decision log `D001`–`D011` (`H004` duplicate `D010` resolved) |
| | [`docs/progress.md`](../docs/progress.md) | Complete chronological milestone log (`M0`–`M4`) & ledger |
| | [`docs/permissions.md`](../docs/permissions.md) | Model licenses, provider usage terms, pricing, and Apple M4 Max audit |
| | [`docs/literature_review.md`](../docs/literature_review.md) | Literature synthesis across 9 core WSP / alignment papers |
| **Handoffs & Independent Audits** | [`docs/handoffs/H001_rq6_competency_pilot.md`](../docs/handoffs/H001_rq6_competency_pilot.md) | `RQ6` synthetic toy-network pilot (`runs/competency_astra/report.md`, `3/3`) |
| | [`docs/handoffs/H002_stage1_awst_t1.md`](../docs/handoffs/H002_stage1_awst_t1.md) | Stage 1 weights-only `C1` + `T1` (`runs/codex_h002/report.md`) |
| | [`docs/handoffs/H003_stage1_awst_t2.md`](../docs/handoffs/H003_stage1_awst_t2.md) | Stage 1 activation-assisted `T2` (`runs/codex_h003/report.md`) |
| | [`docs/handoffs/H004_m3_independent_review.md`](../docs/handoffs/H004_m3_independent_review.md) | Independent M3 statistical/methodology review (`runs/codex_h004/review.md`) |
| **Stored Empirical Outputs (`results/`)** | `m1_stage0_baselines.json`, `m1_remote_reader_validation.json`, `m1_micro_benchmark.json` | Stage 0 baselines (`nothink`/`think`), HTTP `206` reader (`9,892.98x`), M4 Max speed |
| | `m2_rq6_competency_comparison.json`, `m2_conventional_baselines.json`, `m2_h002_verification.json`, `m2_h003_verification.json` | `RQ6` comparison, `C2`/`C3`/`C4` validation, independent `H002`/`H003` verification |
| | `m3_isolated_evaluation.json` | M3 isolated 10-arm confirmatory scorecard, paired CIs, McNemar & Holm statistics |
| **Research Harness (`src/`)** | `src/data/`, `src/weight_io/`, `src/interventions/`, `src/evaluation/`, `src/baselines/`, `src/agent_interface/`, `src/paper/` | Full evaluation, HTTP `206` Safetensors reader, invariant checks, baselines, figures |
| **Isolated Test Logs (External)** | `/Users/kaiwendu/.awst_isolated_eval/` | Hidden test splits & raw M3 logs (`gsm8k_test`, `synthetic_test`, `synthetic_ood_test`, `m3_logs/`) kept strictly outside repo |

---

## 2. Pinned Models & Datasets

- **`Qwen/Qwen3.5-0.8B`** (`Apache 2.0`): revision `2fc06364715b967f1860aea9cf38778875588b17`, primary shard SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (`873,438,784` stored / `852,985,920` active runtime params; `153` vision tensors = `100,592,896` params).
- **`Qwen/Qwen3.5-2B`** (`Apache 2.0`): revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, primary shard SHA-256 `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` (`2,274,069,824` stored params).
- **Datasets (`data/dataset_manifest.json`):** `synthetic_dev` (`900`, `c2e41927...`), `synthetic_val` (`300`, `cd5e3b7f...`), `gsm8k_dev` (`500`, `6e2fbfa1...`), `gsm8k_val` (`200`, `4aa4ef28...`); external isolated `/Users/kaiwendu/.awst_isolated_eval/`: `synthetic_test` (`300`, `c81c7414...`), `synthetic_ood_test` (`100`, `4904d73b...`), `gsm8k_test` (`1319`, `5315f256...`).

---

## 3. Complete Itemized Expense & Compute Summary (M0–M4)

| Item / Session | Mode & Model | Direct Paid USD | Shadow API USD | Scope / Evidence |
|---|---|---:|---:|---|
| Local ML Runs (`M1`–`M4`) | Apple M4 Max MPS (`bfloat16`) | `\$0.00` | `\$0.00` | Baselines, `C2`–`C4`, `H002`/`H003` verification, M3 10-arm isolated eval |
| **`H001` (`RQ6` Competency)** | Codex (`GPT-6.2 Astra`, `max`) | `\$0.00` | `\$1.1002544` | 1 turn, `490,523` tokens; in-turn receipt (`runs/competency_astra/report.md`) |
| **`H002` (Stage 1 `C1` + `T1`)** | Codex (`GPT-6.2 Astra`, `max`) | `\$0.00` | `\$6.041536` | Closed receipt (`runs/codex_h003/accounting.json`); 6.85M in / 50.0K out |
| **`H003` (Stage 1 `T2`)** | Codex (`GPT-6.2 Astra`, `max`) | `\$0.00` | `\$5.988440` | Cutoff receipt (`runs/codex_h003/accounting.json`); 56/64 donor forwards |
| **`H004` (M3 Independent Review)** | Codex (`GPT-6.2 Astra`, `high`) | `\$0.00` | `\$1.161018` | In-turn receipt (`runs/codex_h004/review.md`); 0 model runs |
| **Total Across Initial Study** | **`\$20.00` M2 pilot gate / `\$200.00` ceiling** | **`\$0.00`** | **`\$14.2912484`** | **`\$5.71` unused pilot shadow headroom; `\$0.00` direct spend; `18 GB` cache (`< 30 GB`)** |

*Reconciliation note:* `H002` + `H003` (`\$12.029976`) exceeded the initial `\$12.00` Astra pilot sub-cap by `\$0.029976`, transparently reconciled within the `\$3.00` contingency reserve inside `D005`.

---

## 4. Reproduction Commands

```sh
# 1. Run full automated suite (unit tests + paper/artifact numeric verification)
.venv/bin/pytest -q

# 2. Verify all manuscript claims, citations, figures, and hygiene checks directly
.venv/bin/python src/paper/verify_paper_numbers.py

# 3. Regenerate publication figures strictly from stored JSON outputs
MPLCONFIGDIR=/tmp/mplconfig /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/paper/generate_figures.py

# 4. Recreate frozen H002 (C1, T1) and H003 (T2) edited checkpoints
.venv/bin/python runs/codex_h002/apply_c1.py --recipient-dir hf_cache/models/qwen3.5_0.8b --out-dir hf_cache/edited/h002_recreated_c1 --manifest-path hf_cache/edited/h002_recreated_c1.json
.venv/bin/python runs/codex_h002/apply_t1.py --donor-dir hf_cache/models/qwen3.5_2b --recipient-dir hf_cache/models/qwen3.5_0.8b --out-dir hf_cache/edited/h002_recreated_t1 --manifest-path hf_cache/edited/h002_recreated_t1.json
.venv/bin/python runs/codex_h003/apply_t2.py --donor-dir hf_cache/models/qwen3.5_2b --recipient-dir hf_cache/models/qwen3.5_0.8b --out-dir hf_cache/edited/h003_recreated_t2 --manifest-path hf_cache/edited/h003_recreated_t2.json

# 5. Re-run independent M2 verification & M3 isolated evaluation (requires local hf_cache + /Users/kaiwendu/.awst_isolated_eval)
.venv/bin/python src/evaluation/verify_h002.py
.venv/bin/python src/evaluation/verify_h003.py
.venv/bin/python src/evaluation/run_m3_isolated_eval.py
```

---

## 5. Remaining Future Work & Release Gate

- **Human Gate Pending:** Publication, repository visibility changes, or derived checkpoint distribution require explicit researcher authorization (`AWST_Research_Protocol_v0.3.md` Section 14.2 M4).
- **Deferred / Skipped Scope (`D011`):** Multi-session replication (`N >= 5` independent Codex runs per track), Stage 2 (`Qwen/Qwen3.5-9B`), Stage 3 (`google/gemma-4-E2B-it`), Stage 4 (`moonshotai/Kimi-K3.1-Pro` downstream remote transplantation), blinded manual parse-method audits on isolated completions, and intermediate-scale causal subspace transfer (`0.01%`–`1%` parameter regime).

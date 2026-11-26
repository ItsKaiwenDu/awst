# H003: Stage 1 Activation-Assisted AWST (`T2` — Strictly Separate from `T1`)

- **Date:** 2026-11-26
- **Target Agent / Harness:** GPT-6.2 Astra via Codex
- **Authorization Mode:** `AUTHORIZED IMPLEMENTATION (Academic Research — Stage 1 Track T2 Pilot within $12.00 Astra Cap)`
- **Repository Commit SHA:** `f2c44bb` (or latest `HEAD` on `main`)
- **Model / Checkpoint Revisions:**
  - **Recipient (`C0_raw`):** `Qwen/Qwen3.5-0.8B` (`2fc06364715b967f1860aea9cf38778875588b17`), `hf_cache/models/qwen3.5_0.8b`.
  - **Donor:** `Qwen/Qwen3.5-2B` (`15852e8c16360a2fea060d615a32b45270f8a8fc`), `hf_cache/models/qwen3.5_2b`.
- **Budget / Resource Limits:**
  - Shared `$12.00` Astra pilot cap across `H002` + `H003`.
  - Max `64` donor forward passes strictly on `data/development/` prompts (logged via `AgentWeightToolbox(..., track="T2")`).
  - Max `12` candidate dev evaluation trials on `data/development/` ONLY.

---

## 1. Purpose and Objective

Run the **activation-assisted (`T2`)** Stage 1 AWST experiment, kept **strictly separate** from Track `T1` (`H002`). Unlike `T1`, `T2` permits running `Qwen/Qwen3.5-2B` on `data/development/` prompts to collect internal activations, attention/DeltaNet routing statistics, and causal ablation signals to guide donor-to-recipient transplantation into `Qwen/Qwen3.5-0.8B`.

---

## 2. Measured Track `T2` Conventional Baseline (`C4_act`)

| Arm (`nothink 1024`, greedy) | `carry_add_sub` (`n=50`) | `mult_2x2_3x1` (`n=50`) | `word_2step` (`n=50`) | Synth Overall (`n=150`) | GSM8K Val (`n=100`) | Rel. Frob Norm |
|---|---:|---:|---:|---:|---:|---:|
| **`C0_raw` (`0.8B` base)** | 40.0% | 68.0% | 90.0% | 66.0% | **62.0%** | — |
| **`Qwen3.5-2B` (Donor)** | **78.0%** | **86.0%** | 100.0% | **88.0%** | **82.0%** | — |
| **`C4_act` (`APM + Ridge`, `a=0.02`, 32 dev passes)** | 34.0% | 66.0% | 94.0% | 64.7% | **62.0%** | `0.00441` |

Notice that uniform `C4_act` across layers `8–23` preserved general performance (`62.0%` GSM8K, `94.0%` `word_2step`) but did **not** improve `carry_add_sub` (`34.0%` vs `40.0%` `C0_raw`), and collapsed at `alpha=0.10` (`25.0%` dev). Use donor activations (e.g., error-contrastive activations on problems `2B` solves and `0.8B` fails in `data/development/`) to localize and adapt specific circuits rather than blindly averaging 16 full layers!

---

## 3. Strict Constraints

1. **Data Isolation (CRITICAL):** Use ONLY `data/development/synthetic_dev.jsonl` and `data/development/gsm8k_dev.jsonl` for activation probing and dev selection. **NEVER** inspect or evaluate on `data/validation/` or `/Users/kaiwendu/.awst_isolated_eval/`.
2. **Training-Free & Fixed-Architecture:** Zero gradient descent, backprop, fine-tuning, distillation, or learned adapters. Closed-form projections/ridge/SVD/Procrustes/OT from dev activations ARE permitted. Preserve `873,438,784` params and all `153` frozen vision tensors bit-exact.
3. **Track `T2` Isolation:** Do **not** pool `T2` analysis with `H002`'s `T1` report; keep manifests separate (`runs/codex_h003/`).
4. **Honest Outcome Policy:** Report all failed/reverted trials transparently.

---

## 4. Expected Deliverables

Create directory `runs/codex_h003/` containing:
1. **`runs/codex_h003/apply_t2.py`**:
   - Standalone deterministic Python script exposing:
     `apply_t2(donor_dir, recipient_dir, out_dir, manifest_path, donor_override: dict[str, torch.Tensor] | None = None)`
   - Must support `donor_override` so Antigravity can run the `C5` shuffled-donor control on the exact same pipeline.
2. **`runs/codex_h003/report.md`**:
   - Pre-edit activation localization hypotheses (which layers/components differ between `2B` and `0.8B` on `carry_add_sub` / `gsm8k_dev`)
   - Exact dev trial log (including failed attempts), provenance, `changed_params`, `overall_relative_frobenius_norm`, `donor_forward_passes`, `recipient_forward_passes`, wall time, and token/shadow-cost accounting.

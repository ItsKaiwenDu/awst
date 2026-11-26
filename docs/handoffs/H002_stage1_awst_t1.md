# H002: Stage 1 Weights-Only AWST (`T1`) and Donor-Free Control (`C1`)

- **Date:** 2026-11-26
- **Target Agent / Harness:** GPT-6.2 Astra via Codex
- **Authorization Mode:** `AUTHORIZED IMPLEMENTATION (Academic Research — Stage 1 Pilot within $12.00 Astra Cap)`
- **Repository Commit SHA:** `380b7b9` (or latest M2a-2 `HEAD` on `main`)
- **Model / Checkpoint Revisions:**
  - **Recipient (`C0_raw`):** `Qwen/Qwen3.5-0.8B` (`2fc06364715b967f1860aea9cf38778875588b17`), local path `hf_cache/models/qwen3.5_0.8b` (`873,438,784` params, `153` vision tensors).
  - **Donor:** `Qwen/Qwen3.5-2B` (`15852e8c16360a2fea060d615a32b45270f8a8fc`), local path `hf_cache/models/qwen3.5_2b` (`2,274,069,824` params).
- **Budget / Resource Limits:**
  - `<= $12.00` total shadow/API budget across `H002` + `H003` combined (`$8/M` input, `$0.40/M` cached input, `$40/M` output).
  - Max `12` recipient dev evaluation trials per arm (`C1`, `T1`) on `data/development/` ONLY.
  - Max `200 MB` donor bytes accessed via `AgentWeightToolbox(..., track="T1")`.
  - **ZERO donor forward passes** in `T1`; **ZERO donor access of any kind** in `C1`.

---

## 1. Purpose and Objective

Run two strictly separated Stage 1 weight-space programming arms on `Qwen/Qwen3.5-0.8B` (`nothink`, `max_new_tokens=1024`, greedy `do_sample=False`, `bfloat16`):
1. **`C1` (Donor-Free WSP):** Diagnose mathematical failures and improve `Qwen3.5-0.8B` using **only** recipient weights and `data/development/` probes (no donor access).
2. **`T1` (Weights-Only AWST):** Identify, adapt, and transplant mathematical reasoning weights from `Qwen3.5-2B` into `Qwen3.5-0.8B` using **weights-only** inspection (`AgentWeightToolbox(..., track="T1")`; **no donor forward passes or donor activations**).

Antigravity will independently verify your submitted scripts, check all architectural invariants (`873,438,784` params, `153/153` frozen vision tensors), run `C5` (shuffled-donor version of your `T1` formula) and `float64` reversion controls, and evaluate on the blinded `data/validation/` set.

---

## 2. Verified Architecture & Measured M1/M2a Baselines

### Architecture Comparison (`configs/models/`)
- Both models have **identical** `24` text layers (`3:1` Gated DeltaNet `linear_attention` + `full_attention` at layers `3, 7, 11, 15, 19, 23`), identical `8` attention heads / `2` KV heads / `head_dim=256`, identical `16` linear key & value heads / `head_dim=128`, identical `vocab_size=248320`, and `tie_word_embeddings=true`.
- They differ in `hidden_size` (`1024` vs. `2048`, `2:1` width ratio) and `intermediate_size` (`3584` vs. `6144`).

### Measured Validation Baselines (`nothink 1024`, greedy, M4 Max MPS `bfloat16`)

| Arm | `carry_add_sub` (`n=50`) | `mult_2x2_3x1` (`n=50`) | `word_2step` (`n=50`) | Synth Overall (`n=150`) | GSM8K Val (`n=100`) | Rel. Frob Norm |
|---|---:|---:|---:|---:|---:|---:|
| **`C0_raw` (`0.8B` base)** | 40.0% | 68.0% | 90.0% | 66.0% | **62.0%** | — |
| **`Qwen3.5-2B` (Donor)** | **78.0%** | **86.0%** | 100.0% | **88.0%** | **82.0%** | — |
| **`C2` (Random Gaussian)** | 36.0% | 76.0% | 84.0% | 65.3% | 63.0% | `0.01019` |
| **`C3` (Random `2B` slice, `a=0.05`)** | 40.0% | 64.0% | 78.0% | 60.7% | 55.0% | `0.01116` |
| **`C4_weights` (Procrustes + Hungarian, `a=0.05`)** | 40.0% | 58.0% | 80.0% | 59.3% | 60.0% | `0.01003` |

**Key Diagnostic Insight:** Blanket layer-wide MLP interpolation across layers `8–23` (`C4_weights`) failed to beat `C0_raw` and collapsed at `alpha=0.10` (`17.5%` dev). Smarter surgical localization, attention/DeltaNet + MLP co-adaptation, or low-rank subspace grafting (staying within the training-free, weights-only `T1` rules) is needed if `T1` can work at all.

---

## 3. Strict Constraints and Available Tools

1. **Data Isolation (CRITICAL):**
   - You may read and evaluate ONLY on `data/development/synthetic_dev.jsonl` and `data/development/gsm8k_dev.jsonl`.
   - **NEVER** read, load, or evaluate on `data/validation/` or `/Users/kaiwendu/.awst_isolated_eval/`.
2. **Training-Free & Fixed-Architecture:**
   - Zero backpropagation, gradient descent, Adam/SGD, fine-tuning, distillation, or learned adapters.
   - Forward-only dev measurements on the recipient, closed-form linear algebra (SVD, Procrustes, ridge, spectral/subspace projections), and hypothesis-driven weight edits are permitted.
   - Do **not** alter `config.json`, parameter count (`873,438,784`), shapes, dtypes (`bfloat16`), or any of the `153` vision tensors (`model.visual.*` / `visual.*`).
   - Do **not** overwrite `hf_cache/models/`. Apply all candidate edits via `src/interventions/patch_applier.py:apply_tensor_patches(...)` into temporary directories under `hf_cache/edited/` and clean up discarded candidates.
3. **Track Separation (`C1` vs `T1`):**
   - Use `src/agent_interface/tools.py:AgentWeightToolbox`:
     - For `C1`: `AgentWeightToolbox(donor_dir=None, recipient_dir="hf_cache/models/qwen3.5_0.8b", track="C1")`.
     - For `T1`: `AgentWeightToolbox(donor_dir="hf_cache/models/qwen3.5_2b", recipient_dir="hf_cache/models/qwen3.5_0.8b", track="T1")`.
   - **Never** dump full weight matrices into text/context; inspect via `tensor_stats`, `tensor_svd_summary`, `tensor_slice` (`<=64 KB`), and `layer_alignment_diagnostics`.
4. **Honest Outcome Policy:** Negative results are valid scientific outcomes. If neither `C1` nor `T1` improves `carry_add_sub` / `gsm8k_dev` over `C0_raw` within your trial budget, document the failed attempts and report honestly.

---

## 4. Expected Deliverables

Create directory `runs/codex_h002/` containing:
1. **`runs/codex_h002/apply_c1.py`**:
   - Standalone deterministic Python script with signature/CLI:
     `apply_c1(recipient_dir, out_dir, manifest_path)` -> calls `apply_tensor_patches(...)` and writes `manifest_path`.
2. **`runs/codex_h002/apply_t1.py`**:
   - Standalone deterministic Python script with function:
     `apply_t1(donor_dir, recipient_dir, out_dir, manifest_path, donor_override: dict[str, torch.Tensor] | None = None)`
   - When `donor_override` is passed (by Antigravity for the `C5` shuffled-donor control), your script MUST read donor tensors from `donor_override` instead of `donor_dir` while running the exact same adaptation formula!
3. **`runs/codex_h002/report.md`**:
   - Pre-edit mechanistic hypotheses for `C1` and `T1`
   - Complete dev trial log (including failed/reverted trials, exact dev `carry_add_sub` / `gsm8k_dev` scores)
   - Exact donor tensor provenance, closed-form formulas, `changed_params`, `overall_relative_frobenius_norm`, `donor_bytes_accessed`, `recipient_forward_passes`, wall time, and token/shadow-cost accounting.

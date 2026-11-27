# Project AWST — Progress Log

**Date:** 2026-11-26  
**Current Milestone:** `M2: Agent pilot and main experiments` — **Stage 1 (`T1` `H002` + `T2` `H003`) independently verified (both negative on blinded validation); awaiting Stage 2 Gate decision.**
**Branch:** `main`  
**Stage 1 verified commits:** `477646f` (`H002` verified), `007101b` (`H003` submission) on `main`
**Primary Harness:** Google Antigravity (`Gemini 4 Neon`)  
**Secondary Weight-Space Programmer:** Codex (`GPT-6.2 Astra` — `H001` verified `3/3`; `H002` & `H003` verified)

---

## 1. Budget & Resource Ledger

| Category | Authorized | Spent to Date | Notes |
|---|---:|---:|---|
| Paid Pilot Gate (`D005`) | **\$20.00** (approved for M2) | \$0.00 | M1 local setup & baselines completed at \$0.00; \$13.1302 cumulative shadow (`\$6.87` M2 headroom) |
| Gemini API | \$5.00 (within \$20 pilot / \$30 protocol cap) | \$0.00 | `D001` Option A matched check in M2 |
| Astra API / Codex | \$12.00 (+ \$3.00 `D005` contingency) | \$0.00 direct | `H001` \$1.1003 + `H002` \$6.0415 + `H003` \$5.9884 = **\$13.1302** shadow (`H002`+`H003` = `\$12.0300`, `\$0.0300` reconciled against `D005` \$3.00 contingency). [H003 meter](../runs/codex_h003/accounting.json) |
| Cloud GPU | \$0.00 (protocol cap \$40.00) | \$0.00 | Local M4 Max first |
| Storage & Misc | \$0.00 (protocol cap \$10.00) | \$0.00 | `18G` in `hf_cache/` (`< 30 GB` cap); 580 GiB free |
| **Total Initial Study** | **\$20.00 pilot authorized / \$200.00 hard ceiling** | **\$0.00** | Zero paid work executed |

- **Local Hardware Verified:** Apple M4 Max, 36 GiB unified memory (`38,654,705,664` bytes), 14 CPU cores, macOS `27.2.1` (`26B131`), `580 GiB` free disk space on `/`.
- **Software Gap Identified:** Python `3.14.5` and `3.11` are installed (`huggingface_hub 1.29.0`, `numpy 2.5.1`), but `torch`, `transformers` (`>=5.2`), `safetensors`, `datasets`, `scipy`, and `pytest` are **not installed yet** (deferred to M1 `.venv` setup).

---

## 2. Permissions & Compliance Status

- **Model Licenses:** `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-9B`, and `google/gemma-4-E2B-it` verified **Apache 2.0**. `moonshotai/Kimi-K3.1-Pro` uses custom `kimi-k3.1-pro` license (research/modification allowed; MaaS restrictions; Stage 4 gated, NO download).
- **Academic Scope:** Non-commercial open-weight research disclaimer recorded in `README.md`.
- **Checkpoint Downloads:** Pinned `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` downloaded locally (`7.9G` total in `hf_cache/`).
- **Git Remote:** `origin = https://github.com/ItsKaiwenDu/awst.git`, branch `main` (visibility not modified).

---

## 3. Master Project Checklist

### M0: Literature and research setup
- [x] Read `AWST_Research_Protocol_v0.3.md` completely and inspect repository & hardware (`39a4573`, Apple M4 Max 36 GiB, 580 GiB free).
- [x] Gather relevant primary papers & search for newer late-2026 work (`ropinski2026editready`, `gross2026wspverify`); metadata & URLs indexed in [`docs/literature/README.md`](literature/README.md) (no copyrighted PDFs stored).
- [x] Verify references, read abstracts/methods/limitations, and reassess novelty in [`docs/literature_review.md`](literature_review.md) and [`docs/literature/bibliography.bib`](literature/bibliography.bib).
- [x] Set up `README.md`, repository layout, `.gitignore`, `AGENTS.md`, `GEMINI.md`, `docs/research_protocol.md`, `docs/handoffs/README.md`, `docs/progress.md`, `docs/decisions.md`, and `docs/permissions.md`.
- [x] Verify checkpoint candidates, licenses, provider ToS, live pricing, local compute, and storage in [`docs/permissions.md`](permissions.md); prepare approval gates in [`docs/decisions.md`](decisions.md).
- [x] Define narrow arithmetic diagnostic (`D004`), controls, statistical plan, feasibility criteria, and proposed \$20 pilot (`D005`) in [`docs/preregistration.md`](preregistration.md).

### M1: Baseline and infrastructure *(Completed 2026-11-25 at \$0.00 spend)*
- [x] Set up pinned `.venv` (Python `3.11.16`, `torch 2.15.0`, `transformers 5.24.1`, `safetensors 0.8.0`, `datasets 5.2.0`), `requirements.txt`, `configs/environment.json`, `configs/models/` pinned SHAs (`Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`, `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`), `src/` harness, and `8/8` passing unit tests (`pytest -q`).
- [x] Isolate final held-out test data outside workspace (`/Users/kaiwendu/.awst_isolated_eval/`: `synthetic_test` 300, `synthetic_ood_test` 100, `gsm8k_test` 1319; in-repo dev/val splits recorded in `data/dataset_manifest.json`).
- [x] Validate selective tensor retrieval on `Qwen/Qwen3.5-0.8B` via HTTP `206` `Range` requests (`results/m1_remote_reader_validation.json`: `176,584 / 1,746,942,600` bytes = `9892.98x` reduction, bit-exact match verified).
- [x] Run M4 Max MPS micro-benchmark (`results/m1_micro_benchmark.json`: `109 tok/s` `0.8B`, `64–65 tok/s` `2B`, zero NaNs), reproduce Stage 0 `C0_loopfixed` (`runs/manifests/stage0_c0_loopfixed_patch.json`: `130,188` entries in L15/L19 `down_proj`, `153/153` vision tensors SHA-256 unchanged, `reversion_verified: true`), run validation baselines (`results/m1_stage0_baselines.json`: `nothink 1024` `C0_raw` `62.0%` GSM8K / `40.0%` `carry_add_sub` vs `2B` `82.0%` GSM8K / `78.0%` `carry_add_sub`; `think` check `95% -> 75%` loops without `presence_penalty=1.5`), and freeze [`docs/preregistration.md`](preregistration.md) v1.0.

### M2: Agent pilot and main experiments *(M2a, H001, H002 `T1`, and H003 `T2` all verified; Stage 2 Gate pending)*
- [x] Run approved small competency comparisons and measure actual cost; select programmer(s) using pilot evidence. *(Verified `3/3` for both Gemini 4 Neon and GPT-6.2 Astra in [`results/m2_rq6_competency_comparison.json`](../results/m2_rq6_competency_comparison.json); Astra `$1.10` in-turn shadow / `$0.00` direct API; caveat: `docs/progress.md` disclosed Neon's aggregate `3/3` score prior to Astra run, zero task answers exposed. Confirmed **GPT-6.2 Astra (Codex)** for Stage 1 `H002` & `H003`.)*
- [x] Implement and test conventional transfer (`C4`) and matched controls (`C1`–`C5`); establish immutable patch & provenance logs. *(Evaluated in `results/m2_conventional_baselines.json`, `results/m2_h002_verification.json`, and `results/m2_h003_verification.json`: `C4_weights` `60.0%` GSM8K / `40.0%` `carry_add_sub`; `C4_act` `62.0%` / `34.0%`; `C2` `63.0%` / `36.0%`; `C3` `55.0%` / `40.0%`; `H002` `C5_elem` `56.0%`/`32.0%`, `C5_rowperm` `59.0%`/`40.0%`; `H003` `C5_perm` `57.0%`/`34.0%`, `C5_norm_matched` `59.0%`/`34.0%` vs. `C0_raw` `62.0%` / `40.0%` and `2B` `82.0%` / `78.0%`. `12/12` unit tests pass; `hf_cache` = `18G`.)*
- [x] Complete H002 donor-free C1 and weights-only T1 trials and Antigravity independent verification (`C1`, `T1`, `C5_elem`, `C5_rowperm` on blinded `data/validation/`). *(Dev `+4.0 pp` GSM8K gain did **not** generalize to validation: `T1` `57.0%` GSM8K vs `C0_raw` `62.0%`, `C1` `59.0%`, `C5_rowperm` `59.0%`. See section 8 and [`results/m2_h002_verification.json`](../results/m2_h002_verification.json).)*
- [x] Complete H003 activation-assisted development trials and return the frozen T2 checkpoint. [H003 report](../runs/codex_h003/report.md)
- [x] Independently verify Stage 1 T2, shuffled-donor controls and blinded outcomes. *(Verified negative on validation: `T2` `57.0%` GSM8K / `38.0%` `carry_add_sub` vs `C0_raw` `62.0%`/`40.0%`, `C4_act` `62.0%`/`34.0%`, `C5_perm` `57.0%`/`34.0%`, `C5_norm_matched` `59.0%`/`34.0%`. See section 10 and [`results/m2_h003_verification.json`](../results/m2_h003_verification.json).)*
- [ ] Review donor-dependence, preliminary results, budget, and regressions before approving Stage 2 (`Qwen3.5-9B`).
- [ ] Run approved second donor experiment if justified (Stage 3/4 gated separately).

### M3: Verification and analysis
- [ ] Freeze selected methods and checkpoints before isolated final evaluation.
- [ ] Verify reproducibility from recorded commits/manifests, CIs, independent runs, contamination risks, and alternative explanations.
- [ ] Verify unchanged architecture/parameter count, zero gradient training, provenance, retention, latency, peak memory, and costs.
- [ ] Request independent review without revealing hidden data; record and resolve findings.

### M4: Writing and release preparation
- [ ] Write paper after verified results only, updating abstract to match evidence.
- [ ] Verify all figures, tables, numerical claims, and references against stored outputs.
- [ ] Prepare final artifact manifest, expense summary, and obtain human approval before public release.

---

## 4. Approved Design Decisions (M0–M1 Exit Gates)

1. **Stage 1 Dual-Track Scope (`D008` — APPROVED):** Implement and evaluate both `T1` (weights-only, paired with `C4_weights`) and `T2` (activation-assisted, paired with `C4_act`) strictly unpooled under `README.md` academic research scope.
2. **Checkpoint Downloads & `.venv` Setup (`D007` — APPROVED):** Local `.venv` ML stack installation and pinned `Qwen/Qwen3.5-0.8B` & `Qwen/Qwen3.5-2B` downloads completed (`7.9G` within 30 GB cap).
3. **\$20.00 Pilot Budget Gate (`D005` — APPROVED FOR M2):** Up to \$20.00 authorized for M2 pilot (M1 completed at \$0.00).
4. **Scientific Design (`D001`–`D004`, `D006` — APPROVED):** `D001` Option A (Astra primary + bounded Astra-vs-Neon RQ6 check), `D002` Option A (`nothink 1024` on `C0_raw` per Pilot Protocol v1.0), `D003` (freeze vision tower), `D004` (`carry_add_sub` primary, `mult_2x2_3x1` secondary), `D006` (`/Users/kaiwendu/.awst_isolated_eval/`).

---

## 5. Next actions

1. **Stage 2 Approval Gate (`Qwen3.5-9B -> Qwen3.5-0.8B` vs. Milestone M3/M4):** Both Stage 1 AWST tracks (`T1` `H002` and `T2` `H003`) failed to outperform `C0_raw`, `C4`, or shuffled-donor `C5` controls on blinded `data/validation/` (`57.0%` vs `62.0%` GSM8K; `40.0%` / `38.0%` vs `40.0%` `carry_add_sub`). Pause for researcher decision on whether to skip Stage 2 (`Qwen3.5-9B`) and proceed directly to Milestone M3 isolated final evaluation (`/Users/kaiwendu/.awst_isolated_eval/`) + M4 paper writing, or approve a larger-donor / follow-up experiment.

## 8. Antigravity Independent Verification of H002 (`T1` / `C1` / `C5`), 2026-11-26

- **Verification Runner:** [`src/evaluation/verify_h002.py`](../src/evaluation/verify_h002.py) · **Full Report:** [`results/m2_h002_verification.json`](../results/m2_h002_verification.json)
- **Replay & Invariants:** Independent execution of `apply_c1` and `apply_t1` matched Codex's shard SHA-256 digests bit-for-bit (`b11f16df6de7bfb5f060789477a4e67764946f111feccd3a871d2440227ad11a` and `da9ad05f9c4931477197b46581ed83653485f16763eeb7197bb9aaead2f81414`), `float64` delta reversion succeeded with **`0` differing parameters** (`153/153` frozen vision tensors), and all four verified checkpoints (`C1`, `T1`, `C5_elem`, `C5_rowperm`) preserved `873,438,784` stored parameters, `153/153` frozen vision tensors, `donor_forward_passes: 0`, and `gradient_steps: 0`. Temporary verification checkpoints were cleaned up, keeping `hf_cache/` at **`15G`** (`< 30 GB` cap).
- **Digit-Delta Geometry (`float64`, 10 tied digit rows):** Matched Frobenius norms (`~0.0372`: `C1 = 0.037287`, `T1 = 0.037204`, `C5_elem = 0.037268`, `C5_rowperm = 0.037134`). `T1` is nearly orthogonal to `C1` (`cos = +0.055951`), `C5_elem` (`cos = -0.057268`), and `C5_rowperm` (`cos = -0.003241`), while `C5_elem` and `C5_rowperm` align closely with each other (`cos = +0.889710`) and oppose `C1` (`cos = -0.676905`, `-0.635903`).
- **Blinded Validation Results (`nothink 1024`, greedy, M4 Max MPS `bfloat16`, `~109.4–109.8 tok/s`):**
  - **`C0_raw` (reference):** `62.0%` `gsm8k_val` (`n=100`), `66.0%` `synthetic_val` (`n=150`: `40.0%` `carry_add_sub`, `68.0%` `mult_2x2_3x1`, `90.0%` `word_2step`).
  - **`C4_weights` (reference):** `60.0%` `gsm8k_val`, `59.3%` `synthetic_val` (`40.0%` carry, `58.0%` mult, `80.0%` word).
  - **`C1` (`1/32` donor-free):** `59.0%` `gsm8k_val`, `63.33%` `synthetic_val` (`36.0%` carry, `62.0%` mult, `92.0%` word).
  - **`T1` (`1/32` weights-only):** `57.0%` `gsm8k_val`, `66.67%` `synthetic_val` (`40.0%` carry, `68.0%` mult, `92.0%` word).
  - **`C5_elem` (element-shuffled donor):** `56.0%` `gsm8k_val`, `64.67%` `synthetic_val` (`32.0%` carry, `76.0%` mult, `86.0%` word).
  - **`C5_rowperm` (row-permuted donor):** `59.0%` `gsm8k_val`, `66.0%` `synthetic_val` (`40.0%` carry, `68.0%` mult, `90.0%` word).
- **Paired Inference (Exact McNemar & 10,000-resample Bootstrap 95% CI) & Scientific Finding:**
  - `T1` vs `C0_raw` on `gsm8k_val`: `-5.0 pp` (`[-13.0, +2.0] pp`, `5` improved / `10` regressed, `p = 0.3018`); on `carry_add_sub`: `0.0 pp` (`[-10.0, +10.05] pp`, `4/4`, `p = 1.0`).
  - `T1` vs `C1` on `gsm8k_val`: `-2.0 pp` (`[-8.0, +4.0] pp`, `4/6`, `p = 0.7539`); on `carry_add_sub`: `+4.0 pp` (`[-8.0, +16.0] pp`, `6/4`, `p = 0.7539`).
  - `T1` vs `C5_elem` on `gsm8k_val`: `+1.0 pp` (`[-5.0, +8.0] pp`, `6/5`, `p = 1.0`); on `carry_add_sub`: `+8.0 pp` (`[+2.0, +16.0] pp`, `4/0`, `p = 0.125`).
  - `T1` vs `C5_rowperm` on `gsm8k_val`: `-2.0 pp` (`[-9.0, +5.0] pp`, `5/7`, `p = 0.7744`); on `carry_add_sub`: `0.0 pp` (`[-10.0, +10.0] pp`, `3/3`, `p = 1.0`).
  - `T1` vs `C4_weights` on `gsm8k_val`: `-3.0 pp` (`[-13.0, +7.0] pp`, `11/14`, `p = 0.6900`).
  - **Conclusion for Track `T1` (`H002`):** **Verified negative / non-generalizing result.** The `+4.0 pp` development improvement on `gsm8k_dev` (`+19` on 480-item remainder) did not generalize to `gsm8k_val` (`-5.0 pp` vs `C0_raw`), and `T1` shows zero statistically significant advantage over `C0_raw`, `C1`, `C4_weights`, or shuffled-donor `C5` controls. Under strict Track `T1`/`T2` separation (`D008`), Track `T2` (`H003`) remains pending to test whether activation-assisted localization succeeds where weights-only transfer failed.

## 6. H001 synthetic competency submission, 2026-11-26

- M2 / RQ6 / D001. Codex worked from `5a12b77dcae43ef2e755911015b58c34c25a0e71`. A fresh numerical workspace received the four authorized checkpoints and their suite manifest. The required startup log disclosed the other trial’s aggregate score. [Handoff](handoffs/H001_rq6_competency_pilot.md)
- One invocation of `.venv/bin/python -IB` produced both patches. Reload checks covered tensor keys, shapes, float32 dtypes and affine coefficients for B, and verified the sparse edit and bit-exact reapplication of the corruption for C. All five source hashes and both delivered checkpoint hashes passed `shasum -a 256 -c`. [Solver, report and artifacts](../runs/competency_astra/report.md) · [Per-tensor provenance](../runs/competency_astra/manifest.json)
- One Codex turn (`gpt-6.2-astra`, reasoning effort `max`), zero forward passes, zero training steps and $0.00 in direct API charges. The in-turn reading at 9:53:57 AM PST was 490,523 tokens and $1.1002544 shadow cost against the $4.00 cap. Final usage reconciliation is pending.
- Independently verified `3/3` by Antigravity via `verify_competency_submission` (`task_a_passed: true`, `task_b_agreement: 1.0`, `task_b_rel_l2_error: 0.0`, `task_c_localized: true`, `task_c_agreement: 1.0`); comparison recorded in [`results/m2_rq6_competency_comparison.json`](../results/m2_rq6_competency_comparison.json). The binary `.safetensors` checkpoints remain gitignored locally.

## 7. H002 Stage 1 development pilot, 2026-11-26

- The researcher submitted the authorized H002 brief. Codex recorded the edit formulas, two radii, equal-sized screen and tie-break before editing. Both arms alter ten rows in the tied vocabulary embeddings. C1 selected radius 1/32 before donor inspection. [Design and trial record](../runs/codex_h002/report.md)
- One baseline launch stopped before generation on a runtime-count assertion. The pinned Transformers class omits 15 stored MTP tensors. The corrected check distinguishes 873,438,784 stored entries from 852,985,920 runtime parameters and checks both. The four patch manifests independently record 153 bit-identical vision tensors. [Baseline runtime](../runs/codex_h002/c0_screen.json)
- The C5 override regression and existing intervention, evaluator and weight-reader checks passed, 7 tests. T1 uses only bounded embedding slices with zero donor forwards. All scoring processes load a recipient only. [Override regression](../tests/test_h002_transfer.py) · [Source verification](../runs/codex_h002/source_audit.json)
- Both arms selected radius 1/32. On the full development sets C0 solved 110/300 carry and 264/500 GSM8K, C1 solved 103/300 and 267/500, and T1 solved 105/300 and 286/500. On the disjoint GSM8K remainder T1 corrected 38 base errors and lost 19 base successes. The improvement on GSM8K co-occurred with lower carry accuracy. [Paired results](../runs/codex_h002/dev_results.json)
- The scoring processes generated 1,086,638 tokens across 2,480 prompts in 9,997.74 seconds on local MPS. Each arm used 3/12 candidate evaluations. Two CLI replays regenerated both finalist shards byte for byte. Discarded candidates were removed. Direct API and cloud spend was $0.00. [Resource accounting](../runs/codex_h002/accounting.json) · [Replay evidence](../runs/codex_h002/replay_check.json)
- H002 Codex shadow usage through 1:48:07 PM PST was $5.1043392 across two chat turns at the handoff rates. [Meter and token counts](../runs/codex_h002/accounting.json)
- H002 submission `60286c9` was pushed and matched `origin/main`. Post-push shadow metering through 1:50:25 PM PST was $5.7215696, with $0.00 direct spend. [Meter receipt](../runs/codex_h002/accounting.json)

## 9. H003 activation-assisted pilot, November 26, 2026

- The researcher submitted the H003 implementation brief at starting commit `cb8d828`. Codex froze the twelve-layer head search, causal shortlist, two interpolation strengths and disjoint development slices before the first forward. H002 had closed at $6.041536 shadow cost, superseding its $5.7215696 in-turn receipt. H003 opened with $5.958464 of shared headroom and $0 direct spend. [Design](../runs/codex_h003/design.md)
- Calibration used 32 development prompts per model. Two candidate head ablations used another 24 forwards per model. The signed causal ranking selected layer 13, head 12. The donor's margin increased slightly under ablation, so this result does not establish a necessary arithmetic head. No additional donor forward is required for the edit or its controls. [Localization and margins](../runs/codex_h003/localization.json)
- Both checkpoints modify two recurrent decay scalars. The 0.5 and 1.0 strengths have relative Frobenius norms 0.000176949 and 0.000353898. Both preserved all 873,438,784 stored entries, 153 frozen vision tensors and the loaded runtime signature. Twelve repository tests passed. The development evaluator is scoring the fixed screen and follow-up on local MPS. [Half-strength patch](../runs/codex_h003/half_patch.json) · [Full-strength patch](../runs/codex_h003/full_patch.json)
- Frozen-method commit `939e4ee` was pushed to `origin/main`. The 0.5 finalist scored 23/40 on the screen, compared with 19/40 for strength 1.0. On 64 disjoint prompts per family it solved 27 carry and 38 GSM8K, versus 26 and 36 for C0 and 22 and 36 for C4_act. All four paired 95% intervals include zero. The donor dependence and generalization of this small development difference are unresolved. [Trial log, comparisons and reproduction](../runs/codex_h003/report.md)
- H003 used 56/64 donor forwards, 246,175 recipient forwards including probing, 3/12 candidate trials, four reference evaluations and 245,639 generated tokens. Evaluation took 2,240.48 seconds. The ordinary and identical-value override reproduced the finalist SHA-256. A shuffled source changed the edit and had 13.3 times its norm. Float64 reversion restored the pristine shard digest. All seven development evaluations completed, with $0 direct spend. Codex shadow usage through 2026-11-27T03:04:36.451Z was $5.988440. Independent C5 behavioral scoring and blinded validation remain with Antigravity. [Replay](../runs/codex_h003/replay.json) · [Resources](../runs/codex_h003/accounting.json)
- The first fresh meter after the resumed chat showed $12.029976 combined shadow usage at 7:04:36 PM PST on November 26, $0.029976 above the shared $12 cap. The development runs and local verification had already finished. The final results are in the working tree. The final results commit and push are pending a budget decision. The published localization commit is `939e4ee`. The meter is a cutoff snapshot.

## 10. Antigravity Independent Verification of H003 (`T2` / `C5`), 2026-11-26

- **Verification Runner:** [`src/evaluation/verify_h003.py`](../src/evaluation/verify_h003.py) · **Full Report:** [`results/m2_h003_verification.json`](../results/m2_h003_verification.json)
- **Replay & Architectural Invariants:** Independent execution of `apply_t2` matched Codex's `h003_half` shard SHA-256 bit-for-bit (`ea5c744fe4d6627c2faefee13e2c2ad97c44f3ad012c7adc491d60290f6bb732`), as did `donor_override` with exact donor vectors under an inaccessible `donor_dir` sentinel. `float64` delta reversion restored the pristine shard SHA-256 with **`0` differing parameters** (`153/153` frozen vision tensors). All three verified checkpoints (`T2`, `C5_perm`, `C5_norm_matched`) preserved `873,438,784` stored parameters, `153/153` frozen vision tensors, `changed_params: 2` (`layers.13.linear_attn.{A_log,dt_bias}` head `12`), `donor_forward_passes: 0`, and `gradient_steps: 0`. Temporary verification directories were cleaned up, keeping `hf_cache/` at **`18G`** (`< 30 GB` cap).
- **2D Delta Geometry (`layers.13.linear_attn.{A_log,dt_bias}` head 12):** `T2` delta = `[-0.10546875, +0.12890625]` (`||delta||_2 = 0.16655`, `rel_frob = 0.00017695`); canonical `C5_perm` (`seed=42`, mapped donor head `4`) reproduced Codex's shuffled shard SHA-256 `f47af66b18e02bf65920f47b750616cf2301da02c9a1ea31cb9972471926708c` with `[-0.03125, -2.21875]` (`||delta||_2 = 2.21897`, **`13.32x`** `T2` norm, `cos = -0.7650` vs `T2`). Norm-matched `C5_norm_matched` scaled the same shuffled direction to `[-0.00235558, -0.1640625]` (`||delta||_2 = 0.16408`, **`0.9851x`** `T2` norm, `1.49%` `bfloat16` rounding difference, `cos = -0.7648` vs `T2`).
- **Blinded Validation Results (`nothink 1024`, greedy, M4 Max MPS `bfloat16`, `~104.8–106.2 tok/s`):**
  - **`C0_raw` (reference):** `62.0%` `gsm8k_val` (`n=100`), `66.0%` `synthetic_val` (`n=150`: `40.0%` `carry_add_sub`, `68.0%` `mult_2x2_3x1`, `90.0%` `word_2step`).
  - **`C4_act` (reference):** `62.0%` `gsm8k_val`, `64.67%` `synthetic_val` (`34.0%` carry, `66.0%` mult, `94.0%` word).
  - **`T2` (`strength=0.5`, L13 H12):** `57.0%` `gsm8k_val`, `66.0%` `synthetic_val` (`38.0%` carry, `66.0%` mult, `94.0%` word).
  - **`C5_perm` (`13.32x` shuffled-donor):** `57.0%` `gsm8k_val`, `66.67%` `synthetic_val` (`34.0%` carry, `78.0%` mult, `88.0%` word).
  - **`C5_norm_matched` (`0.985x` shuffled-donor):** `59.0%` `gsm8k_val`, `65.33%` `synthetic_val` (`34.0%` carry, `72.0%` mult, `90.0%` word).
- **Paired Inference (Exact McNemar & 10,000-resample Bootstrap 95% CI) & Stage 1 Conclusion:**
  - `T2` vs `C0_raw` on `gsm8k_val`: `-5.0 pp` (`[-11.0, +1.0] pp`, `2` improved / `7` regressed, `p = 0.1797`); on `carry_add_sub`: `-2.0 pp` (`[-10.0, +6.0] pp`, `2/3`, `p = 1.0`).
  - `T2` vs `C4_act` on `gsm8k_val`: `-5.0 pp` (`[-15.0, +5.0] pp`, `10/15`, `p = 0.4244`); on `carry_add_sub`: `+4.0 pp` (`[-10.0, +18.0] pp`, `7/5`, `p = 0.7744`).
  - `T2` vs `C5_perm` on `gsm8k_val`: `0.0 pp` (`[-11.0, +11.0] pp`, `15/15`, `p = 1.0`); on `carry_add_sub`: `+4.0 pp` (`[-10.0, +18.0] pp`, `8/6`, `p = 0.7905`).
  - `T2` vs `C5_norm_matched` on `gsm8k_val`: `-2.0 pp` (`[-8.0, +4.0] pp`, `4/6`, `p = 0.7539`); on `carry_add_sub`: `+4.0 pp` (`[-6.0, +14.0] pp`, `4/2`, `p = 0.6875`).
  - **Budget Reconciliation (`D005`):** Codex's `H003` final meter (`\$5.988440` shadow) plus `H002` (`\$6.041536`) totaled `\$12.029976`, exceeding the `\$12.00` Astra sub-cap by `\$0.029976`. That `\$0.03` overrun is reconciled transparently against the approved `\$3.00` contingency buffer inside the `\$20.00` M2 pilot gate (`D005`), with `\$0.00` direct API/cloud spend (`\$13.1302` total Codex shadow across `H001`–`H003`).
  - **Scientific Conclusion for Stage 1 (`Qwen3.5-2B -> Qwen3.5-0.8B`):** **Both Track `T1` (`H002`) and Track `T2` (`H003`) are verified negative on blinded validation.** Neither agent-directed weights-only nor activation-assisted training-free transplantation outperformed `C0_raw`, `C4`, or shuffled-donor `C5` controls on `gsm8k_val` or `carry_add_sub`. Ready for Stage 2 Gate review (`D005` / Protocol Section 14.2 M2).

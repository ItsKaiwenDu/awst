# Project AWST — Progress Log

**Date:** 2026-11-26  
**Current Milestone:** `M2: Agent pilot and main experiments` — **H002 independently verified (negative on blinded validation); H003 (`T2`) awaiting researcher submission.**
**Branch:** `main`  
**H002 verified commit:** `f1b6e4a` on `main`
**Primary Harness:** Google Antigravity (`Gemini 4 Neon`)  
**Secondary Weight-Space Programmer:** Codex (`GPT-6.2 Astra` — `H001` verified `3/3`; `H002` verified; awaiting `H003`)

---

## 1. Budget & Resource Ledger

| Category | Authorized | Spent to Date | Notes |
|---|---:|---:|---|
| Paid Pilot Gate (`D005`) | **\$20.00** (approved for M2) | \$0.00 | M1 local setup & baselines completed at \$0.00 |
| Gemini API | \$5.00 (within \$20 pilot / \$30 protocol cap) | \$0.00 | `D001` Option A matched check in M2 |
| Astra API / Codex | \$12.00 (within \$20 pilot / \$60 protocol cap) | \$0.00 direct | `H001` \$1.1003 + `H002` \$5.7216 = \$6.8218 shadow (`\$6.2784` left in \$12.00 shared `H002`+`H003` Astra cap). [H002 metering](../runs/codex_h002/accounting.json) |
| Cloud GPU | \$0.00 (protocol cap \$40.00) | \$0.00 | Local M4 Max first |
| Storage & Misc | \$0.00 (protocol cap \$10.00) | \$0.00 | `15G` in `hf_cache/` (`< 30 GB` cap); 580 GiB free |
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

### M2: Agent pilot and main experiments *(M2a & H001 complete; H002 `T1` independently verified; H003 `T2` pending)*
- [x] Run approved small competency comparisons and measure actual cost; select programmer(s) using pilot evidence. *(Verified `3/3` for both Gemini 4 Neon and GPT-6.2 Astra in [`results/m2_rq6_competency_comparison.json`](../results/m2_rq6_competency_comparison.json); Astra `$1.10` in-turn shadow / `$0.00` direct API; caveat: `docs/progress.md` disclosed Neon's aggregate `3/3` score prior to Astra run, zero task answers exposed. Confirmed **GPT-6.2 Astra (Codex)** for Stage 1 `H002` & `H003`.)*
- [x] Implement and test conventional transfer (`C4`) and matched controls (`C1`–`C5`); establish immutable patch & provenance logs. *(Evaluated in `results/m2_conventional_baselines.json` and `results/m2_h002_verification.json`: `C4_weights` `60.0%` GSM8K / `40.0%` `carry_add_sub`; `C4_act` `62.0%` / `34.0%`; `C2` `63.0%` / `36.0%`; `C3` `55.0%` / `40.0%`; `C5_elem` `56.0%` / `32.0%`; `C5_rowperm` `59.0%` / `40.0%` vs. `C0_raw` `62.0%` / `40.0%` and `2B` `82.0%` / `78.0%`. `12/12` unit tests pass; `hf_cache` = `15G`.)*
- [x] Complete H002 donor-free C1 and weights-only T1 trials and Antigravity independent verification (`C1`, `T1`, `C5_elem`, `C5_rowperm` on blinded `data/validation/`). *(Dev `+4.0 pp` GSM8K gain did **not** generalize to validation: `T1` `57.0%` GSM8K vs `C0_raw` `62.0%`, `C1` `59.0%`, `C5_rowperm` `59.0%`. See section 8 and [`results/m2_h002_verification.json`](../results/m2_h002_verification.json).)*
- [ ] Run the separately authorized H003 activation-assisted pilot (`T2`) after researcher submission and independently verify Stage 1 `T2` controls and blinded outcomes. [H003 handoff](handoffs/H003_stage1_awst_t2.md)
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

1. **Pause for manual researcher submission of `H003` (`T2`, activation-assisted AWST):** [H003 handoff](handoffs/H003_stage1_awst_t2.md) (remaining balance `\$6.28` under the `\$12.00` shared Astra cap).
2. Once `runs/codex_h003/` returns, Antigravity will independently verify `T2` invariants, `float64` reversion, `C5` shuffled-donor control, and blinded `data/validation/` evaluation, then review Stage 1 results before the Stage 2 (`Qwen3.5-9B`) gate.

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

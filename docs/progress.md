# Project AWST — Progress Log

**Date:** 2026-11-26  
**Current Milestone:** `M2: Agent pilot and main experiments` — **M2a Complete; Paused for Manual Codex Handoffs (`H001`–`H003`)**  
**Branch:** `main`  
**Latest Commit SHA:** `380b7b9` (`Commit: Add M2 Agent Toolkit, Competency Suite & Baselines`, pushed to `origin/main`)  
**Primary Harness:** Google Antigravity (`Gemini 4 Neon`)  
**Secondary Weight-Space Programmer:** Codex (`GPT-6.2 Astra` — manual handoffs `H001`, `H002`, `H003` ready)

---

## 1. Budget & Resource Ledger

| Category | Authorized | Spent to Date | Notes |
|---|---:|---:|---|
| Paid Pilot Gate (`D005`) | **\$20.00** (approved for M2) | \$0.00 | M1 local setup & baselines completed at \$0.00 |
| Gemini API | \$5.00 (within \$20 pilot / \$30 protocol cap) | \$0.00 | `D001` Option A matched check in M2 |
| Astra API / Codex | \$12.00 (within \$20 pilot / \$60 protocol cap) | \$0.00 | Ready for M2 handoffs (`H001`, `H002`) |
| Cloud GPU | \$0.00 (protocol cap \$40.00) | \$0.00 | Local M4 Max first |
| Storage & Misc | \$0.00 (protocol cap \$10.00) | \$0.00 | Local 580 GiB free |
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

### M2: Agent pilot and main experiments *(M2a Complete at \$0.00 spend; M2b awaiting Codex `H001`–`H003`)*
- [ ] Run approved small competency comparisons and measure actual cost; select programmer(s) using pilot evidence. *(Gemini 4 Neon isolated trial verified `3/3` in `results/m2_rq6_competency_neon.json`; GPT-6.2 Astra brief ready at [`docs/handoffs/H001_rq6_competency_pilot.md`](handoffs/H001_rq6_competency_pilot.md).)*
- [x] Implement and test conventional transfer (`C4`) and matched controls (`C1`–`C5`); establish immutable patch & provenance logs. *(Evaluated in `results/m2_conventional_baselines.json`: `C4_weights` `60.0%` GSM8K / `40.0%` `carry_add_sub`; `C4_act` `62.0%` / `34.0%`; `C2` `63.0%` / `36.0%`; `C3` `55.0%` / `40.0%` vs. `C0_raw` `62.0%` / `40.0%` and `2B` `82.0%` / `78.0%`. `11/11` unit tests pass; `hf_cache` = `11G`.)*
- [ ] Run approved `Qwen3.5-2B -> Qwen3.5-0.8B` AWST trials; log all outcomes including failures. *(Codex handoff briefs ready: [`docs/handoffs/H002_stage1_awst_t1.md`](handoffs/H002_stage1_awst_t1.md) for `C1`+`T1` and [`docs/handoffs/H003_stage1_awst_t2.md`](handoffs/H003_stage1_awst_t2.md) for `T2`.)*
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

## 5. Exact Next Action (Paused for Researcher Codex Handoffs `H001`–`H003`)

1. Researcher submits [`docs/handoffs/H001_rq6_competency_pilot.md`](handoffs/H001_rq6_competency_pilot.md), [`docs/handoffs/H002_stage1_awst_t1.md`](handoffs/H002_stage1_awst_t1.md), and [`docs/handoffs/H003_stage1_awst_t2.md`](handoffs/H003_stage1_awst_t2.md) to **GPT-6.2 Astra in Codex**.
2. Once Codex returns its scripts/reports, Antigravity independently verifies invariants (`873,438,784` params, `153/153` frozen vision tensors), runs `C5` shuffled-donor and `float64` reversion controls, and evaluates all candidate checkpoints (`C1`, `T1`, `T2`) on `data/validation/` on M4 Max MPS.

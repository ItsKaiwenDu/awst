# Project AWST — Progress Log

**Date:** 2026-11-25  
**Current Milestone:** `M0: Literature and research setup` — **Completed; Ready for Exit Gate Review & Human Approvals**  
**Branch:** `main`  
**Base Commit SHA (at start of M0):** `39a4573`  
**Primary Harness:** Google Antigravity (`Gemini 4 Neon`)  
**Secondary Weight-Space Programmer:** Codex (`GPT-6.2 Astra` — manual handoff only; not yet invoked)

---

## 1. Budget & Resource Ledger

| Category | Authorized | Spent to Date | Notes |
|---|---:|---:|---|
| Paid Pilot Gate (`D005`) | **\$0.00** (proposed \$20.00) | \$0.00 | **Awaiting explicit researcher approval** |
| Gemini API | \$0.00 (protocol cap \$30.00) | \$0.00 | No paid API calls made |
| Astra API / Codex | \$0.00 (protocol cap \$60.00) | \$0.00 | No Codex handoffs submitted |
| Cloud GPU | \$0.00 (protocol cap \$40.00) | \$0.00 | Local M4 Max first |
| Storage & Misc | \$0.00 (protocol cap \$10.00) | \$0.00 | Local 580 GiB free |
| **Total Initial Study** | **\$0.00 authorized / \$200.00 hard ceiling** | **\$0.00** | Zero paid work executed |

- **Local Hardware Verified:** Apple M4 Max, 36 GiB unified memory (`38,654,705,664` bytes), 14 CPU cores, macOS `27.2.1` (`26B131`), `580 GiB` free disk space on `/`.
- **Software Gap Identified:** Python `3.14.5` and `3.11` are installed (`huggingface_hub 1.29.0`, `numpy 2.5.1`), but `torch`, `transformers` (`>=5.2`), `safetensors`, `datasets`, `scipy`, and `pytest` are **not installed yet** (deferred to M1 `.venv` setup).

---

## 2. Permissions & Compliance Status

- **Model Licenses:** `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-9B`, and `google/gemma-4-E2B-it` verified **Apache 2.0**. `moonshotai/Kimi-K3.1-Pro` uses custom `kimi-k3.1-pro` license (research/modification allowed; MaaS restrictions; Stage 4 gated, NO download).
- **Provider Terms of Service Gate:** **UNRESOLVED / BLOCKED FOR WEIGHT-IMPROVEMENT RUNS.** Both OpenAI Terms of Use and Google Gemini API Additional Terms prohibit using outputs/services to develop competing models (`docs/permissions.md` Section 2, `D008`).
- **Checkpoint Downloads:** **NOT STARTED / PENDING APPROVAL.**
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

### M1: Baseline and infrastructure *(Blocked pending M0 Human Approval Gate)*
- [ ] Set up pinned `.venv` (`torch` MPS, `transformers>=5.2`, `safetensors`, `datasets`, `pytest`), reproducible configs, weight reader, evaluation harness, and unit tests.
- [ ] Isolate final held-out test data outside workspace (`/Users/kaiwendu/.awst_isolated_eval/`); verify scoring, tokenization, thinking-mode config, and baseline reproducibility (`C0_raw` & `C0_loopfixed`). Freeze pilot protocol and checkpoint revisions.
- [ ] Validate selective tensor retrieval on small Safetensors files via HTTP `Range` (`206 Partial Content`); measure download bytes, storage, and M4 Max throughput. Replicate permitted WSP baseline or document failure/block.

### M2: Agent pilot and main experiments
- [ ] Run approved small competency comparisons and measure actual cost; select programmer(s) using pilot evidence.
- [ ] Implement and test conventional transfer (`C4`) and matched controls (`C1`–`C5`); establish immutable patch & provenance logs.
- [ ] Run approved `Qwen3.5-2B -> Qwen3.5-0.8B` AWST trials; log all outcomes including failures.
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

## 4. Pending Human Approvals (M0 Exit Gate)

Before M1 begins, researcher review and approval are required on:
1. **Provider Terms Compliance (`D008` / `docs/permissions.md`):** Resolve competing-model clause for OpenAI Codex (`GPT-6.2 Astra`) and Google (`Gemini 4`) before any actual LLM weight-improvement experiment (Stage 0 WSP replication, `C1`, `T1`, `T2`), or designate an open-weight fallback programmer.
2. **Checkpoint Downloads & `.venv` Setup (`D007`):** Authorize installing ML packages into `.venv` and downloading pinned `Qwen/Qwen3.5-0.8B` (~1.75 GB) and `Qwen/Qwen3.5-2B` (~4.25 GB) within the 30 GB local cache cap.
3. **\$20.00 Pilot Budget Gate (`D005`):** Authorize up to \$20.00 paid spend for M2 competency + Stage 1 pilot (local M1 work remains \$0.00).
4. **Scientific Design Choices (`D001`, `D002`, `D003`, `D004`, `D006`):**
   - `D001`: Keep small matched Astra-vs-Neon competency check for RQ6 (Option A) vs. drop RQ6 (Option B)?
   - `D002`: Evaluate Stage 1 primarily on `C0_loopfixed` (after Stage 0 loop-repair check) to eliminate the Takehi et al. loop-repair confound?
   - `D003`: Freeze multimodal vision encoder; focus AWST on text backbone?
   - `D004` & `D006`: Approve 3-tier synthetic arithmetic diagnostic and `/Users/kaiwendu/.awst_isolated_eval/` external test-isolation path?

---

## 5. Exact Next Action (Once Approved)

1. Create `.venv` on M4 Max with PyTorch MPS + `transformers>=5.2` + `safetensors` + `pytest`, and benchmark `Qwen3.5-0.8B` & `Qwen3.5-2B` on a 10-problem micro-batch.
2. Pin exact Hugging Face commit SHAs for `Qwen3.5-0.8B` and `Qwen3.5-2B` in `configs/models/`.
3. Build `src/data/`, `src/evaluation/`, and `src/weight_io/` (including HTTP `206` Safetensors Range reader tests) and isolate frozen test splits externally.

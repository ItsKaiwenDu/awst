# Research and Engineering Decision Log

Statuses: `SETTLED` (agreed in protocol or verified fact), `PROPOSED / NEEDS APPROVAL` (awaiting researcher sign-off at M0 Exit Gate), `DEFERRED` (gated behind later milestones).

---

### D001 — Agent Division of Labor vs. RQ6 Comparison
- **Status:** `APPROVED — Option A (2026-11-25)`
- **Context:** The launch instructions designate **GPT-6.2 Astra via Codex** as the primary weight-space programmer for core AWST weight transplantation experiments, while **Antigravity (Gemini 4 Neon)** handles research coordination, infrastructure, experimental design, baselines/controls, independent verification, documentation, and paper writing. Meanwhile, `AWST_Research_Protocol_v0.3.md` includes `RQ6` (matched competency/cost comparison of Astra vs. Neon) and Section 14.2 M2 calls for small competency comparisons before final selection.
- **Options:**
  - **Option A (Recommended):** Keep Gemini 4 Neon as coordinator/verifier/infra/paper lead and GPT-6.2 Astra (Codex) as primary AWST programmer, **and** run one small, strictly bounded matched competency comparison (~5–10 synthetic/diagnostic items, ~\$4–\$6 total) in early M2 to preserve empirical evidence for `RQ6`.
  - **Option B:** Skip the Astra-vs-Neon competency comparison altogether, pre-assign GPT-6.2 Astra as sole AWST programmer (subject to ToS), mark `RQ6` as `dropped / out-of-scope`, and reallocate the \$30 Gemini API budget to reserve.
- **Recommendation:** **Option A**, operating under the non-commercial academic research scope stated in `README.md`.
- **2026-11-26 H001 submission.** Codex returned its three synthetic-task answers and two checkpoints. Programmer selection remains open pending independent scoring and cost reconciliation. [H001 report](../runs/competency_astra/report.md)

---

### D002 — Disentangling Qwen3.5-0.8B Loop Repair from Math Transfer (Critical Scientific Challenge)
- **Status:** `APPROVED — Option A (2026-11-25)`
- **Context:** Literature verification (`takehi2026sparks`) confirmed that raw `Qwen3.5-0.8B` suffers from severe reasoning loops, and Takehi et al. raised its GSM8K accuracy from `13.8%` to `67.1%` (and MATH-500 `7.9% -> 48.1%`) by subtracting a single direction across `131,072` MLP weights (`0.016%`) in layers 15 and 19.
- **Scientific Risk:** If Stage 1 (`Qwen3.5-2B -> Qwen3.5-0.8B`) starts from raw `Qwen3.5-0.8B` (`C0_raw`), any intervention that incidentally disrupts the loop mechanism could produce a massive apparent gain without transferring any mathematical reasoning from the `2B` donor. Even worse, `C1` (donor-free WSP) could saturate the easy loop-repair gains (`~67%`), leaving little signal unless loop repair and math transfer are explicitly separated.
- **Options:**
  - **Option A (Recommended):** In Stage 0, establish **both** `C0_raw` (unmodified `Qwen3.5-0.8B`) and `C0_loopfixed` (loop-suppressed `Qwen3.5-0.8B`, reproducing Takehi et al.'s `0.016%` edit or verifying thinking/non-thinking prompt settings). Run Stage 1 primary AWST trials starting from **`C0_loopfixed`** as the clean recipient baseline (where loop failure is already controlled), while also recording `C0_raw` and `C1` on `C0_raw` as reference points.
  - **Option B:** Run Stage 1 only on `C0_raw` and rely solely on `C1` (donor-free WSP) at matched edit budget to control for loop repair.
- **Recommendation:** **Option A.** Also check in M1 whether `Qwen3.5-0.8B` exhibits different loop rates under standard vs. non-thinking chat templates so prompt configuration is frozen cleanly before M2.
- **Empirical M1 Addendum (2026-11-25):** Stage 0 replication cleanly built `C0_loopfixed` (`exp8_math/runs/p50c/edit`, bit-exact `base_sha256` & `edited_sha256` verified). Under local HF `generate` (`enable_thinking=True, temp=0.6, max_new_tokens=2048`, without vLLM `presence_penalty=1.5` or `32K` token budget), `p50c` reduced `0.8B` GSM8K looping from `95% -> 75%` (`25% -> 30%` pass@1, `100%` truncated at 2048). Meanwhile, **`enable_thinking=False` (`max_new_tokens=1024`, greedy)** naturally eliminated loops on unmodified `C0_raw` (`2.0%` GSM8K loops, `62.0%` pass@1) while `p50c` slightly hurt `nothink` accuracy (`56.0%` GSM8K, `30.0%` `carry_add_sub` vs `40.0%` raw). Therefore, Pilot Protocol v1.0 freezes **`nothink 1024` greedy on `C0_raw`** as the primary Stage 1 baseline (cleanly loop-free, ~5x faster, and +20 pp GSM8K / +38 pp `carry_add_sub` donor lead), retaining `C0_loopfixed` as Stage 0 reference.

---

### D003 — Hybrid Multimodal Architecture Scope for Qwen3.5
- **Status:** `APPROVED (2026-11-25)`
- **Context:** M0 verification showed `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` are both `Qwen3_5ForConditionalGeneration` multimodal checkpoints containing a vision encoder alongside a 24-layer hybrid text backbone (`3:1` Gated DeltaNet linear attention + Full Attention, `full_attention_interval: 4`, SwiGLU MLPs).
- **Decision Proposal:**
  1. Treat the **vision tower as strictly frozen** across all math experiments (verify bit-exact SHA-256 unchanged after every edit).
  2. Restrict candidate AWST target tensors to the text language model backbone (embedding/LM head, Gated DeltaNet layers, Full Attention layers, SwiGLU MLP `gate_proj`/`up_proj`/`down_proj`, and RMSNorms), logging the exact layer type (`linear_attention`, `full_attention`, `mlp`, `norm`, `embed`) for every edit.
  3. Keep `vocab_size` and tied/untied embedding invariants explicitly verified on save/reload.

---

### D004 — Narrow Synthetic Arithmetic Diagnostic Specification
- **Status:** `APPROVED (2026-11-25)`
- **Context:** Protocol Section 5A & 9 call for a narrow, measurable grade-school arithmetic skill verified to be neither at ceiling nor floor before freezing.
- **Proposal:** Construct a deterministic, seeded generator (`src/data/`) covering three graded sub-skills:
  - **Sub-skill A (Multi-digit carry/borrow arithmetic):** 3- to 5-digit addition and subtraction requiring >=2 carries/borrows.
  - **Sub-skill B (Multi-digit multiplication):** 2x2-digit and 3x1-digit integer multiplication.
  - **Sub-skill C (2-step arithmetic word problems):** Template-varied word problems using operations from A/B, with **disjoint held-out phrasings/templates** reserved for OOD generalization evaluation.
- **Split Design:** Fixed disjoint `60/20/20` split generated with separate seeds: e.g., `300` dev / `100` validation / `100` isolated test per sub-skill (plus `100` isolated OOD template items). Exact difficulty tier selected for M2 pilot will be frozen in M1 after checking `C0_loopfixed` (`0.8B`) is in the non-degenerate range (`15%–80%`) and `Qwen3.5-2B` leads by `>= 15` percentage points.
- **Empirical M1 Addendum (2026-11-25):** On `synthetic_val.jsonl` (`nothink 1024`, `n=50`/subskill): `carry_add_sub` (`40.0%` `0.8B_raw` vs `78.0%` `2B`, **+38.0 pp**) is frozen as the primary synthetic diagnostic; `mult_2x2_3x1` (`68.0%` vs `86.0%`, **+18.0 pp**) is frozen as secondary; `word_2step` (`90.0%` vs `100.0%`) is at ceiling and demoted to sanity/retention checking only.

---

### D005 — Proposed \$20.00 Pilot Plan and Spending Gate
- **Status:** `APPROVED FOR M2 (2026-11-25; M1 local setup remains $0.00)`
- **Context:** Protocol v0.3 enforces a hard \$200 initial study ceiling and requires explicit authorization for a \$20 pilot before any paid work.
- **Proposed \$20.00 Allocation:**
  - **M1 local setup, Stage 0 baselines, synthetic eval, and Safetensors HTTP Range validation on M4 Max:** **\$0.00** (local compute).
  - **GPT-6.2 Astra (Codex / API) pilot cap:** **\$12.00** (covers 1 small competency/weight-reading check + 2–3 bounded Stage 1 `2B -> 0.8B` pilot runs with strict per-run token/turn caps; Codex subscription usage also shadow-costed at API rates `\$8/\$40` per 1M tokens).
  - **Gemini 4 Neon pilot cap:** **\$5.00** (only if `D001` Option A matched competency check is approved; otherwise \$0.00).
  - **Cloud GPU:** **\$0.00** (use local M4 Max; pause and ask if throughput is too slow).
  - **Pilot contingency buffer:** **\$3.00**.
- **Stop Rule:** Halt immediately if cumulative actual/shadow spend reaches \$20.00, or if Stage 0/M1 finds no usable donor-recipient gap.

---

### D006 — Hidden Evaluation Data Isolation
- **Status:** `APPROVED (2026-11-25)`
- **Context:** Protocol Section 12 requires hidden tests and labels to live **outside** the agent-readable workspace, not merely in a folder named `test/`.
- **Proposal:** Store frozen GSM8K test and synthetic test splits in `/Users/kaiwendu/.awst_isolated_eval/` (outside `/Users/kaiwendu/Documents/GitHub/awst`), with `data/test/` and `*hidden_test*` also gitignored as defense in depth. Only the researcher (or an approved frozen evaluator invocation during M3) runs evaluations against that directory; neither Antigravity nor Codex reads files inside `.awst_isolated_eval/` during M1–M2.

---

### D007 — M1 Local Runtime Environment on Apple M4 Max
- **Status:** `APPROVED (2026-11-25)`
- **Context:** M0 audit found Apple M4 Max (36 GiB unified memory, 580 GiB free disk), Python 3.14.5, and Python 3.11, with **no** `torch`, `transformers`, `safetensors`, `mlx`, or `pytest` installed yet. `Qwen3.5` requires `transformers >= 5.2.0` (with Gated DeltaNet support).
- **Proposal:** In M1 (after approval), create a gitignored `.venv` using a Python version with verified binary wheels for PyTorch MPS + `transformers >= 5.2` + `safetensors` + `datasets` + `scipy` + `pytest`. Run a 10-problem micro-benchmark on `Qwen3.5-0.8B` and `Qwen3.5-2B` before running full dev/val evaluations.

---

### D008 — Stage 1 Dual-Track Scope (Weights-Only `T1` vs. Activation-Assisted `T2`)
- **Status:** `APPROVED (2026-11-25)`
- **Context:** Literature review (`goworek2026replicating`) highlighted that frontier LLMs often rely on shallow input-layer weight cues when donor execution is unavailable.
- **Decision:** Implement and evaluate **both** `T1` (weights-only, zero donor forward passes, paired with `C4_weights`) and `T2` (activation-assisted on `data/development/`, paired with `C4_act`) in Stage 1, strictly unpooled and under separate manifests.

### D009 — H002 within-arm development selection

The researcher submitted the H002 implementation handoff on 2026-11-26. Codex fixed two equal pre-round edit radii per arm and screened each on the existing first 20 carry and 20 GSM8K development problems. Total correct selects the finalist, with ties going to the smaller radius. The remaining 280 carry and 480 GSM8K development items are reserved for measuring each frozen choice and the common unedited recipient. The validation and isolated-test plan remains governed by the frozen protocol. [H002 design](../runs/codex_h002/report.md)

### D010 — H003 activation localization and development selection

The researcher submitted the H003 implementation handoff on November 26, 2026. Within the authorized T2 arm, Codex froze the head-matching rule, two-head ablation shortlist and two strengths before any model forward. The pilot reserves 16 calibration, 20 selection and 64 disjoint follow-up examples from each development family. Equal-family total correct, then carry correct, then smaller strength selects one edit. C0 and the archived C4 activation baseline are measured on the same development prompts. [Frozen design](../runs/codex_h003/design.md)

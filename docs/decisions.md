# Research and Engineering Decision Log

Statuses: `SETTLED` (agreed in protocol or verified fact), `PROPOSED / NEEDS APPROVAL` (awaiting researcher sign-off at M0 Exit Gate), `DEFERRED` (gated behind later milestones).

---

### D001 — Agent Division of Labor vs. RQ6 Comparison
- **Status:** `PROPOSED / NEEDS APPROVAL`
- **Context:** The launch instructions designate **GPT-6.2 Astra via Codex** as the primary weight-space programmer for core AWST weight transplantation experiments, while **Antigravity (Gemini 4 Neon)** handles research coordination, infrastructure, experimental design, baselines/controls, independent verification, documentation, and paper writing. Meanwhile, `AWST_Research_Protocol_v0.3.md` includes `RQ6` (matched competency/cost comparison of Astra vs. Neon) and Section 14.2 M2 calls for small competency comparisons before final selection.
- **Options:**
  - **Option A (Recommended):** Keep Gemini 4 Neon as coordinator/verifier/infra/paper lead and GPT-6.2 Astra (Codex) as primary AWST programmer, **and** run one small, strictly bounded matched competency comparison (~5–10 synthetic/diagnostic items, ~\$4–\$6 total) in early M2 to preserve empirical evidence for `RQ6`.
  - **Option B:** Skip the Astra-vs-Neon competency comparison altogether, pre-assign GPT-6.2 Astra as sole AWST programmer (subject to ToS), mark `RQ6` as `dropped / out-of-scope`, and reallocate the \$30 Gemini API budget to reserve.
- **Recommendation:** **Option A**, provided ToS compliance (`D008`) is cleared for both or for the tested tasks (note: testing on synthetic toy networks or non-LLM diagnostic tasks avoids LLM weight-improvement ToS ambiguity).

---

### D002 — Disentangling Qwen3.5-0.8B Loop Repair from Math Transfer (Critical Scientific Challenge)
- **Status:** `PROPOSED / NEEDS APPROVAL`
- **Context:** Literature verification (`takehi2026sparks`) confirmed that raw `Qwen3.5-0.8B` suffers from severe reasoning loops, and Takehi et al. raised its GSM8K accuracy from `13.8%` to `67.1%` (and MATH-500 `7.9% -> 48.1%`) by subtracting a single direction across `131,072` MLP weights (`0.016%`) in layers 15 and 19.
- **Scientific Risk:** If Stage 1 (`Qwen3.5-2B -> Qwen3.5-0.8B`) starts from raw `Qwen3.5-0.8B` (`C0_raw`), any intervention that incidentally disrupts the loop mechanism could produce a massive apparent gain without transferring any mathematical reasoning from the `2B` donor. Even worse, `C1` (donor-free WSP) could saturate the easy loop-repair gains (`~67%`), leaving little signal unless loop repair and math transfer are explicitly separated.
- **Options:**
  - **Option A (Recommended):** In Stage 0, establish **both** `C0_raw` (unmodified `Qwen3.5-0.8B`) and `C0_loopfixed` (loop-suppressed `Qwen3.5-0.8B`, reproducing Takehi et al.'s `0.016%` edit or verifying thinking/non-thinking prompt settings). Run Stage 1 primary AWST trials starting from **`C0_loopfixed`** as the clean recipient baseline (where loop failure is already controlled), while also recording `C0_raw` and `C1` on `C0_raw` as reference points.
  - **Option B:** Run Stage 1 only on `C0_raw` and rely solely on `C1` (donor-free WSP) at matched edit budget to control for loop repair.
- **Recommendation:** **Option A.** Also check in M1 whether `Qwen3.5-0.8B` exhibits different loop rates under standard vs. non-thinking chat templates so prompt configuration is frozen cleanly before M2.

---

### D003 — Hybrid Multimodal Architecture Scope for Qwen3.5
- **Status:** `PROPOSED / NEEDS APPROVAL`
- **Context:** M0 verification showed `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` are both `Qwen3_5ForConditionalGeneration` multimodal checkpoints containing a vision encoder alongside a 24-layer hybrid text backbone (`3:1` Gated DeltaNet linear attention + Full Attention, `full_attention_interval: 4`, SwiGLU MLPs).
- **Decision Proposal:**
  1. Treat the **vision tower as strictly frozen** across all math experiments (verify bit-exact SHA-256 unchanged after every edit).
  2. Restrict candidate AWST target tensors to the text language model backbone (embedding/LM head, Gated DeltaNet layers, Full Attention layers, SwiGLU MLP `gate_proj`/`up_proj`/`down_proj`, and RMSNorms), logging the exact layer type (`linear_attention`, `full_attention`, `mlp`, `norm`, `embed`) for every edit.
  3. Keep `vocab_size` and tied/untied embedding invariants explicitly verified on save/reload.

---

### D004 — Narrow Synthetic Arithmetic Diagnostic Specification
- **Status:** `PROPOSED / NEEDS APPROVAL`
- **Context:** Protocol Section 5A & 9 call for a narrow, measurable grade-school arithmetic skill verified to be neither at ceiling nor floor before freezing.
- **Proposal:** Construct a deterministic, seeded generator (`src/data/`) covering three graded sub-skills:
  - **Sub-skill A (Multi-digit carry/borrow arithmetic):** 3- to 5-digit addition and subtraction requiring >=2 carries/borrows.
  - **Sub-skill B (Multi-digit multiplication):** 2x2-digit and 3x1-digit integer multiplication.
  - **Sub-skill C (2-step arithmetic word problems):** Template-varied word problems using operations from A/B, with **disjoint held-out phrasings/templates** reserved for OOD generalization evaluation.
- **Split Design:** Fixed disjoint `60/20/20` split generated with separate seeds: e.g., `300` dev / `100` validation / `100` isolated test per sub-skill (plus `100` isolated OOD template items). Exact difficulty tier selected for M2 pilot will be frozen in M1 after checking `C0_loopfixed` (`0.8B`) is in the non-degenerate range (`15%–80%`) and `Qwen3.5-2B` leads by `>= 15` percentage points.

---

### D005 — Proposed \$20.00 Pilot Plan and Spending Gate
- **Status:** `PROPOSED / NEEDS APPROVAL`
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
- **Status:** `PROPOSED / NEEDS APPROVAL`
- **Context:** Protocol Section 12 requires hidden tests and labels to live **outside** the agent-readable workspace, not merely in a folder named `test/`.
- **Proposal:** Store frozen GSM8K test and synthetic test splits in `/Users/kaiwendu/.awst_isolated_eval/` (outside `/Users/kaiwendu/Documents/GitHub/awst`), with `data/test/` and `*hidden_test*` also gitignored as defense in depth. Only the researcher (or an approved frozen evaluator invocation during M3) runs evaluations against that directory; neither Antigravity nor Codex reads files inside `.awst_isolated_eval/` during M1–M2.

---

### D007 — M1 Local Runtime Environment on Apple M4 Max
- **Status:** `PROPOSED / NEEDS APPROVAL`
- **Context:** M0 audit found Apple M4 Max (36 GiB unified memory, 580 GiB free disk), Python 3.14.5, and Python 3.11, with **no** `torch`, `transformers`, `safetensors`, `mlx`, or `pytest` installed yet. `Qwen3.5` requires `transformers >= 5.2.0` (with Gated DeltaNet support).
- **Proposal:** In M1 (after approval), create a gitignored `.venv` using a Python version with verified binary wheels for PyTorch MPS + `transformers >= 5.2` + `safetensors` + `datasets` + `scipy` + `pytest`. Run a 10-problem micro-benchmark on `Qwen3.5-0.8B` and `Qwen3.5-2B` before running full dev/val evaluations.

---

### D008 — Provider Terms of Service Compliance Gate
- **Status:** `BLOCKED / NEEDS RESEARCHER RESOLUTION`
- **Context:** Both OpenAI Terms of Use and Google Gemini API Additional Terms prohibit using outputs/services to develop competing models (`docs/permissions.md` Section 2).
- **Action Required:** Before authorizing any Stage 0 WSP replication (`takehi2026sparks`) or AWST weight-editing run (`C1`, `T1`, `T2`) using GPT-6.2 Astra or Gemini 4, the researcher must either (1) confirm permissible academic/non-competing status or institutional agreement, (2) record an explicit compliance determination, or (3) select a permissively licensed open-weight fallback programmer.

# Provisional Study Preregistration (DRAFT v0.1 — NOT YET FROZEN)

> [!WARNING]
> **Draft status:** This preregistration is prepared at Milestone M0 for researcher review. **Exact checkpoint commit SHAs, synthetic arithmetic difficulty parameters, baseline prompt/thinking-mode settings, and confirmatory sample sizes will ONLY be frozen after M1 baseline verification and before opening any held-out test set.**

---

## 1. Research Questions and Hypotheses

- **RQ1 (Primary):** Can AI agents selectively identify, extract, adapt, and transplant grade-school mathematical reasoning capabilities from a larger pretrained donor (`Qwen3.5-2B`, then `Qwen3.5-9B` if justified) into a smaller recipient (`Qwen3.5-0.8B`) without gradient-based training or changing architecture/parameter count?
- **RQ2:** Does agent-directed transplantation (`T1`/`T2`) outperform conventional algorithm-directed transfer (`C4`) under matched resource constraints?
- **RQ3:** Does access to a stronger donor (`T1`/`T2`) provide measurable benefit beyond donor-free AI weight editing of the recipient alone (`C1`)?
- **RQ4 (Stage 3, gated):** Can the method generalize across model families (`Qwen3.5-9B -> Gemma 4 E2B`)?
- **RQ5 (Stage 4, optional gated):** Can bounded weights-only remote access support useful transfer without downloading the full donor?
- **RQ6 (Exploratory, pending `D001`):** How do permitted agents (GPT-6.2 Astra vs. Gemini 4 Neon) compare per unit of cost and compute on diagnostic weight-space tasks?

**Primary Hypothesis (`H1`):** On frozen held-out evaluation, AWST (`T1` in weights-only track, or `T2` evaluated separately in activation-assisted track) achieves higher exact-answer pass@1 than both donor-free AI editing (`C1`) and a matched conventional donor-transfer baseline (`C4`), while preserving recipient architecture and parameter count.  
**Null Hypothesis (`H0`):** Expected difference `T1 - C1 <= 0` or `T1 - C4 <= 0` on the primary endpoint.

**Secondary Hypotheses:**
- **H2 (Generalization & Mechanism):** Gains persist on unseen problems/OOD templates and are not solely explained by reduced repetition, truncation, or answer-format changes.
- **H3 (Alignment):** Alignment-aware donor adaptation outperforms unaligned direct slice copying (`C3`).
- **H4 (Selective Remote Access, Stage 4 gated):** Selective tensor retrieval achieves measurable gains using substantially fewer donor bytes than a full checkpoint download.

---

## 2. Staged Models and Track Separation

- **Stage 0 (Baseline & Loop Repair):** `Qwen/Qwen3.5-0.8B` (pinned HF revision TBD in M1). Produces two reference states: `C0_raw` (unmodified) and `C0_loopfixed` (reproducing Takehi et al.'s loop-suppression edit, `D002`).
- **Stage 1 (Primary Feasibility):** Donor `Qwen/Qwen3.5-2B` -> Recipient `Qwen/Qwen3.5-0.8B` (evaluated cleanly on `C0_loopfixed`, with `C0_raw` comparison recorded to avoid loop-repair confounding).
- **Stage 2 (Gated):** `Qwen/Qwen3.5-9B` -> `Qwen/Qwen3.5-0.8B` (approved only if Stage 1 justifies scaling and budget permits).
- **Tracks kept strictly separate:**
  - **Track `T1` (Weights-Only):** Agent inspects donor metadata, summary statistics, and bounded tensor slices; **donor forward passes forbidden**. Compared against a weights-only `C4` alignment baseline.
  - **Track `T2` (Activation-Assisted, separate):** Donor calibration activations permitted. Compared against activation-guided `C4` (`APM` / `Transport and Merge`). Results are never pooled with `T1`.

---

## 3. Experimental Conditions and Matched Resource Accounting

| Arm | Description | Controls For |
|---|---|---|
| `C0` (`C0_raw`, `C0_loopfixed`) | Unmodified recipient & loop-suppressed recipient | Reference baselines & loop-repair separation |
| `C1` | Donor-free AI weight editing on recipient only | Generic WSP / recipient self-repair |
| `C2` | Random perturbations on recipient (matched edit count & Frobenius norm) | Nonspecific regularization/noise effects |
| `C3` | Random compatible donor tensor slices (matched transfer budget) | Informed donor component selection |
| `C4` | Conventional algorithmic donor transfer (weights-only alignment for `T1`; APM/OT-merge for `T2`) | Value of agentic control over algorithmic baselines |
| `C5` | AWST transformation applied to shuffled/unrelated donor values (matched shape & scale) | Genuine donor information vs. structural/scale artifact |
| `T1` | Agent-directed donor selection & adaptation (weights-only) | Primary AWST track |
| `T2` | Agent-directed donor selection & adaptation (activation-assisted, optional/separate) | Activation-assisted AWST track |

**Mandatory per-run logged budgets:** edited tensor list, total edited parameter count, relative Frobenius edit norm (`||ΔW||_F / ||W||_F`), candidate trials evaluated on dev, recipient/donor forward passes, generated model tokens, agent input/output/reasoning tokens, donor bytes accessed, wall-clock time, and exact USD cost (including Codex shadow API cost).

---

## 4. Datasets, Isolation, and Evaluation Rules

- **Pilot / Secondary Diagnostic:** Seeded synthetic arithmetic suite (`D004`: carry/borrow arithmetic, multiplication, 2-step word problems with held-out OOD templates; `60/20/20` dev/val/test split).
- **Primary Confirmatory Benchmark:** **GSM8K** test set (or reproducibly sampled subset frozen prior to M3 if local compute/budget requires; official GSM8K train split only for dev/val).
- **Retention & Regressions:** **IFEval** (instruction following) + held-out general-language QA/perplexity sample.
- **Isolation:** Frozen test splits reside in `/Users/kaiwendu/.awst_isolated_eval/` outside the workspace (`D006`). No test questions, answers, or test-derived optimization signals may enter agent context or dev loops.
- **Decoding & Scoring:** Identical prompt template, chat/thinking setting, temperature (`0.0` greedy for primary deterministic pass@1), `max_new_tokens`, and stop criteria across all arms within a stage. Deterministic exact-answer parser + logged metrics for:
  1. `pass@1` (primary; incorrect, missing, and truncated scored as incorrect)
  2. Completion rate, repetition/looping rate, truncation rate, parser-invalid rate
  3. Matched-token-budget accuracy diagnostic and completed-only secondary diagnostic
  4. Blinded manual audit of a sample of parsed outputs.

---

## 5. Statistical Plan and Provisional Success Criteria

- **Replicates:** Target `>= 3` independent agent sessions in the pilot and `>= 5` independent sessions per key arm (`C1`, `C4`, `C5`, `T1`) in confirmatory M2/M3 runs, refined after pilot variance estimation.
- **Inference:** Paired effect sizes (percentage-point difference) with **95% confidence intervals** via paired/hierarchical bootstrap clustering by agent run and problem. Holm-Bonferroni correction across the two primary confirmatory contrasts (`T1 vs C1` and `T1 vs C4`). Supplementary McNemar's test for single-checkpoint paired comparisons.
- **Reversion Check:** For any winning `T1`/`T2` checkpoint, revert the transplanted delta (`W_reverted = W_edit - ΔW`) and verify performance returns to `C0` (plus matched random perturbation control).

| Criterion | Provisional Threshold |
|---|---|
| Mathematical gain | `>= +5` pp over `C0` (`C0_loopfixed`) on primary endpoint |
| Agent & donor advantage | Statistically supported positive advantage over `C1` and `C4` (Holm-adjusted) |
| Donor dependence | Meaningful advantage over `C5`; gain vanishes upon reversion |
| Generalization | Positive transfer on unseen OOD synthetic templates |
| Capability retention | `<= 3` pp drop on IFEval / general retention check |
| Structural invariant | Bit-exact identical parameter count, shapes, runtime graph, frozen vision tower, 0 gradient steps |
| Device efficiency | Peak memory and latency on M4 Max within `10%` of base model |

---

## 6. \$20.00 Pilot Gate and Go / No-Go Criteria

- **Pilot Budget:** Up to **\$20.00** total (`D005`).
- **Go Criteria to Advance Past Pilot:**
  1. Stage 0 establishes reproducible `C0_raw` and `C0_loopfixed` baselines on M4 Max within acceptable latency.
  2. Donor (`Qwen3.5-2B`) outperforms `C0_loopfixed` (`Qwen3.5-0.8B`) by `>= 15` pp on at least one non-ceiling/non-floor synthetic skill and on GSM8K validation.
  3. Selected programmer demonstrates basic weight-inspection competency and produces valid, shape-preserving edits without systemic harness failure within the \$20 pilot cap.
- **No-Go / Pivot Outcome:** If the pilot shows zero donor advantage, broken weight-space competence, or prohibitive cost/latency, pause, report the negative/mixed pilot data faithfully, and consult the researcher before spending beyond \$20.

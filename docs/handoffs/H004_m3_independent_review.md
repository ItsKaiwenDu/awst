# H004: Milestone M3 Independent Methodology & Statistical Review (REVIEW-ONLY)

- **Date:** 2026-11-27
- **Target Agent / Harness:** GPT-6.2 Astra via Codex
- **Authorization Mode:** `REVIEW-ONLY (Zero Weight Edits, Zero New Model Runs, Zero Hidden Test Data Access)`
- **Repository Commit SHA:** Latest `HEAD` on `main` (post-M3 evaluation commit)
- **Model / Checkpoint Revisions:** N/A (read-only audit of code, manifests, and aggregate JSON results)
- **Budget / Resource Limits:**
  - Max **`$2.50` shadow cost** (within remaining `$6.87` headroom of `D005`'s `$20.00` M2 pilot gate; `$0.00` direct API spend).
  - **Zero** GPU/MPS model loads, zero forward passes, zero edits to `src/` or checkpoints.
  - **Strict Isolation:** NEVER access `/Users/kaiwendu/.awst_isolated_eval/`. Audit ONLY in-repo code, manifests, and aggregate JSON outputs.

---

## 1. Purpose and Objective

Per `AWST_Research_Protocol_v0.3.md` Section 14.2 (Milestone M3): *"Request an independent review (Codex if permitted or another approved reviewer) without revealing hidden data; record and resolve findings. If evidence is insufficient, label the study exploratory or negative."*

Conduct a skeptical, publication-grade peer review of Project AWST's completed M0–M3 experimental pipeline and negative confirmatory results before Milestone M4 manuscript drafting.

---

## 2. Files to Audit (Read-Only)

1. Protocol & Pre-Evaluation Freeze:
   - `AWST_Research_Protocol_v0.3.md`
   - `docs/preregistration.md` (including Section 8 M3 freeze addendum)
   - `docs/decisions.md` (`D001`–`D010`)
   - `docs/progress.md`
2. Evaluation & Statistical Code:
   - `src/evaluation/run_m3_isolated_eval.py`
   - `src/evaluation/retention.py`
   - `src/evaluation/verify_h002.py` and `src/evaluation/verify_h003.py`
   - `tests/test_retention.py`
3. Machine-Readable Aggregate Results (no hidden test items):
   - `results/m1_stage0_baselines.json`
   - `results/m1_remote_reader_validation.json`
   - `results/m2_rq6_competency_comparison.json`
   - `results/m2_conventional_baselines.json`
   - `results/m2_h002_verification.json`
   - `results/m2_h003_verification.json`
   - `results/m3_isolated_evaluation.json`

---

## 3. Specific Review Questions

Please evaluate and report candid findings on:
1. **Statistical Validity & Multiple Comparisons:** Are the paired bootstrap 95% CIs, exact two-sided McNemar tests, and step-down Holm-Bonferroni adjustments in `run_m3_isolated_eval.py` mathematically sound? Are any subtle statistical pitfalls present (e.g., `n=200` power vs `+5 pp` threshold, `N=1` agent session per track)?
2. **Isolation, Contamination, and Alternative Explanations:** Are there any subtle leakage channels, parser artifacts, truncation/looping confounds, or public-benchmark pretraining contamination risks that must be explicitly disclosed in M4?
3. **Fairness of Baselines & Controls:** Are `C0`–`C5` fairly compared against `T1` and `T2`? Is our interpretation of `T1 vs C4_weights` (`+7.5 pp`, raw `p=0.040`, Holm `p=0.080`, driven by `C4_weights` degrading `C0_raw` while `C5_T1_rowperm` matches `T1`) and `H3` (`C3` `-20 pp` OOD degradation) completely defensible?
4. **Scope & Claim Calibration for M4 Paper:** Which exact claims are strongly supported by M0–M3 evidence, and which claims must be softened or scoped as exploratory limitations?

---

## 4. Expected Deliverable

Write a concise, candid review memo to `runs/codex_h004/review.md` (with structured severity tags: `[BLOCKING]`, `[MAJOR]`, `[MINOR]`, `[SUGGESTION]`, plus token/shadow-cost accounting). Do not modify code or evaluation outputs.

# Frozen Pilot Protocol v1.0 (2026-11-25)

> [!IMPORTANT]
> **Status: FROZEN PILOT PROTOCOL v1.0** — Frozen after Milestone M1 baseline & invariant verification, prior to opening any isolated test set in `/Users/kaiwendu/.awst_isolated_eval/`. Any post-freeze change to primary endpoints, models, or decoding settings requires a versioned protocol amendment in `docs/decisions.md` and `docs/preregistration.md` (labeling any affected prior runs exploratory).

---

## 1. Research Questions and Hypotheses

- **RQ1 (Primary):** Can AI agents selectively identify, extract, adapt, and transplant grade-school mathematical reasoning capabilities from a larger pretrained donor (`Qwen/Qwen3.5-2B`, then `Qwen/Qwen3.5-9B` if justified) into a smaller recipient (`Qwen/Qwen3.5-0.8B`) without gradient-based training or changing architecture/parameter count?
- **RQ2:** Does agent-directed transplantation (`T1`/`T2`) outperform conventional algorithm-directed transfer (`C4`) under matched resource constraints?
- **RQ3:** Does access to a stronger donor (`T1`/`T2`) provide measurable benefit beyond donor-free AI weight editing of the recipient alone (`C1`)?
- **RQ4 (Stage 3, gated):** Can the method generalize across model families (`Qwen/Qwen3.5-9B -> google/gemma-4-E2B-it`)?
- **RQ5 (Stage 4, optional gated):** Can bounded weights-only remote access support useful transfer without downloading the full donor?
- **RQ6 (Exploratory, `D001` Option A):** How do permitted agents (`GPT-6.2 Astra` vs. `Gemini 4 Neon`) compare per unit of cost and compute on diagnostic weight-space tasks?

**Primary Hypothesis (`H1`):** On frozen held-out evaluation, AWST (`T1` in weights-only track; `T2` evaluated separately in activation-assisted track) achieves higher exact-answer `pass@1` than both donor-free AI editing (`C1`) and a matched conventional donor-transfer baseline (`C4`), while preserving recipient architecture and parameter count.  
**Null Hypothesis (`H0`):** Expected difference `T1 - C1 <= 0` or `T1 - C4 <= 0` on the primary endpoint.

**Secondary Hypotheses:**
- **H2 (Generalization & Mechanism):** Gains persist on unseen problems/OOD templates and are not solely explained by reduced repetition, truncation, or answer-format changes.
- **H3 (Alignment):** Alignment-aware donor adaptation outperforms unaligned direct slice copying (`C3`).
- **H4 (Selective Remote Access):** Selective tensor retrieval achieves measurable gains using substantially fewer donor bytes than a full checkpoint download (M1 verified `9,892.98x` byte reduction on `Qwen/Qwen3.5-0.8B` via HTTP `206`).

---

## 2. Pinned Checkpoints and Verified Architecture Compatibility

| Role | Repo ID | Pinned HF Revision | Primary Shard SHA-256 | Total Params | Text Backbone Params | Vision Params (Frozen) |
|---|---|---|---|---:|---:|---:|
| **Recipient (`C0_raw`)** | `Qwen/Qwen3.5-0.8B` (Apache 2.0) | `2fc06364715b967f1860aea9cf38778875588b17` | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` | `873,438,784` | `772,845,888` | `100,592,896` (153 tensors) |
| **Stage 0 Loop-Fix Ref (`C0_loopfixed`)** | `Qwen/Qwen3.5-0.8B` + `exp8_math/runs/p50c/edit` | `2fc06364715b967f1860aea9cf38778875588b17` | Deterministic patch (`130,188` bf16 entries in L15/L19 `down_proj`, rel Frob `0.002379`, bit-exact reversion verified) | `873,438,784` | `772,845,888` | `100,592,896` (153/153 SHA-256 unchanged) |
| **Stage 1 Donor** | `Qwen/Qwen3.5-2B` (Apache 2.0) | `15852e8c16360a2fea060d615a32b45270f8a8fc` | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` | `2,274,069,824` | `1,942,653,248` | `331,416,576` |

**Verified Structural Compatibility (`configs/models/`):**
- Both checkpoints share **identical** depth (`24` text layers, `3:1` Gated DeltaNet `linear_attention` + `full_attention`), `num_attention_heads=8`, `num_key_value_heads=2`, `head_dim=256`, `linear_conv_kernel_dim=4`, `linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_num_key_heads=16`, `linear_num_value_heads=16`, `vocab_size=248320`, `tie_word_embeddings=true`, and `bfloat16` storage.
- They differ in `hidden_size` (`1024` vs. `2048`, exact `2:1` width ratio) and `intermediate_size` (`3584` vs. `6144`).
- **Scope (`D003`):** Vision tower is strictly frozen (all 153 vision tensors must pass bit-exact SHA-256 verification after every edit).

---

## 3. Measured Stage 0 & Stage 1 Baselines (Apple M4 Max MPS, `bfloat16`)

### 3.1 Primary Operating Mode (`enable_thinking=False`, Greedy `do_sample=False`, `max_new_tokens=1024`)

| Model | Synthetic Overall (`n=150`) | `carry_add_sub` (`n=50`) | `mult_2x2_3x1` (`n=50`) | `word_2step` (`n=50`) | GSM8K Val (`n=100`) | Loop Rate (Synth / GSM8K) | Speed (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`Qwen3.5-0.8B` (`C0_raw`)** | **66.0%** | **40.0%** | **68.0%** | 90.0% | **62.0%** | 6.0% / 2.0% | 109.4–109.8 |
| **`Qwen3.5-0.8B` (`C0_loopfixed`)** | **61.3%** | **30.0%** | **62.0%** | 92.0% | **56.0%** | 2.0% / 1.0% | 110.0 |
| **`Qwen3.5-2B` (Donor)** | **88.0%** | **78.0%** | **86.0%** | 100.0% | **82.0%** | 6.7% / 2.0% | 63.4–63.9 |
| **Donor Lead (`2B - C0_raw`)** | **+22.0 pp** | **+38.0 pp** | **+18.0 pp** | +10.0 pp (ceiling) | **+20.0 pp** | — | — |

### 3.2 Stage 0 `think`-Mode Replication Reference (`enable_thinking=True`, `max_new_tokens=2048`)
- Greedy (`n=10–12` dev micro-benchmark): `0.8B_raw` hits `90.0%–91.7%` loops; `0.8B_loopfixed` still hits `80.0%–91.7%` loops; `2B` hits `25.0%–40.0%` loops.
- Seeded sampling (`n=20` GSM8K val; `temp=0.6, top_p=0.95, top_k=20, seed=42`, without vLLM `presence_penalty=1.5` and `32K` token budget): `0.8B_raw` `pass@1 = 25.0%` (`95.0%` loops, `100%` trunc) vs. `0.8B_loopfixed` `pass@1 = 30.0%` (`75.0%` loops, `100%` trunc).
- **Conclusion (`D002` Empirical Addendum):** Takehi et al.'s `p50c` edit reduces `think`-mode looping (`95% -> 75%`) but relies on `presence_penalty=1.5` and `32K` context for its full `67.1%` GSM8K jump, and slightly hurts `nothink` accuracy (`62% -> 56%`). By contrast, **`enable_thinking=False` (`max_new_tokens=1024`, greedy)** naturally eliminates the reasoning-loop confound (`2.0%` GSM8K loops on `C0_raw`), runs ~5x faster locally, and preserves clean **+20.0 pp** GSM8K and **+38.0 pp** `carry_add_sub` gaps.

---

## 4. Frozen Stage 1 Experimental Configuration

- **Primary Recipient Base:** `Qwen/Qwen3.5-0.8B` (`C0_raw`, revision `2fc06364715b967f1860aea9cf38778875588b17`).
- **Frozen Inference Config:** `enable_thinking=False`, `do_sample=False` (greedy), `max_new_tokens=1024`, `bfloat16` on Apple M4 Max MPS, identical chat prompt (`src/evaluation/runner.py`).
- **Frozen Narrow Arithmetic Diagnostic (`D004`):**
  - Primary sub-skill: **`carry_add_sub`** (`C0_raw = 40.0%`, `2B = 78.0%`, `+38.0 pp` gap).
  - Secondary sub-skill: **`mult_2x2_3x1`** (`C0_raw = 68.0%`, `2B = 86.0%`, `+18.0 pp` gap).
  - Retention/sanity check only: `word_2step` (`90.0%` vs `100.0%`, at ceiling).
- **Frozen Confirmatory Benchmark:** `gsm8k` (`C0_raw = 62.0%`, `2B = 82.0%`, `+20.0 pp` gap).
- **Frozen Dataset Checksums (`data/dataset_manifest.json`):**
  - In-repo dev/val: `synthetic_dev.jsonl` (`900`, `c2e41927...`), `synthetic_val.jsonl` (`300`, `cd5e3b7f...`), `gsm8k_dev.jsonl` (`500`, `6e2fbfa1...`), `gsm8k_val.jsonl` (`200`, `4aa4ef28...`).
  - Isolated external (`/Users/kaiwendu/.awst_isolated_eval/`): `synthetic_test.jsonl` (`300`, `c81c7414...`), `synthetic_ood_test.jsonl` (`100`, `4904d73b...`), `gsm8k_test.jsonl` (`1319`, `5315f256...`).
- **Pilot Sample Sizes (M2):** Screen candidates on `dev`; validate finalists on `n=50` `carry_add_sub` + `n=50` `mult_2x2_3x1` + `n=100` `gsm8k_val`. Reserve `/Users/kaiwendu/.awst_isolated_eval/` strictly for frozen M3 confirmatory evaluation (`synthetic_test`, `synthetic_ood_test`, and `gsm8k_test` full `1319` items or a preregistered power-verified subset if multi-arm runtimes require).

---

## 5. Experimental Arms, Track Separation, and Budget Accounting

| Arm | Description | Controls For |
|---|---|---|
| `C0` (`C0_raw`) | Unmodified `Qwen/Qwen3.5-0.8B` (`nothink 1024`) | Primary baseline |
| `C1` | Donor-free AI weight editing on `C0_raw` | Generic WSP / self-repair |
| `C2` | Random perturbations on `C0_raw` (matched edit count & Frobenius norm) | Nonspecific noise/regularization |
| `C3` | Random compatible `Qwen3.5-2B` donor tensor slices (matched transfer budget) | Informed donor component selection |
| `C4` | Conventional algorithmic donor transfer (weights-only SVD/alignment for `T1`; APM / OT-merge for `T2`) | Advantage over algorithmic transfer |
| `C5` | AWST transformation applied to shuffled/unrelated donor values (matched shape & scale) | Genuine donor information vs. scale/structure |
| `T1` | Agent-directed donor selection & adaptation (weights-only; **no donor forward passes**) | Primary AWST track |
| `T2` | Agent-directed donor selection & adaptation (activation-assisted, **evaluated separately**) | Activation-assisted extension |

**Tracks `T1` and `T2` are never pooled.** Every run logs edited tensor names, changed parameter count, relative Frobenius norm, dev trials evaluated, forward passes, generated tokens, agent input/output/reasoning tokens, donor bytes accessed, wall time, and exact USD cost (including Codex shadow API cost).

---

## 6. Statistical Analysis Plan and Success Criteria

- **Replicates:** Target `>= 3` independent agent sessions in the M2 pilot and `>= 5` independent sessions per key arm (`C1`, `C4`, `C5`, `T1`) in confirmatory runs.
- **Inference:** Paired effect sizes (percentage-point difference) with **95% confidence intervals** via paired/hierarchical bootstrap clustering by agent run and problem. Holm-Bonferroni correction across the two primary contrasts (`T1 vs C1` and `T1 vs C4`). Supplementary McNemar's test for single-checkpoint paired comparisons.
- **Reversion Check:** Revert winning `T1`/`T2` edits in `float64` (`verify_exact_reversion`) and verify gain disappears.

| Criterion | Provisional Confirmatory Threshold |
|---|---|
| Mathematical gain | `>= +5` pp over `C0_raw` on primary GSM8K endpoint (and positive gain on `carry_add_sub`) |
| Agent & donor advantage | Statistically supported advantage over `C1` and `C4` (Holm-adjusted) |
| Donor dependence | Meaningful advantage over `C5`; gain vanishes upon reversion |
| Generalization | Positive transfer on unseen `synthetic_ood_test` templates |
| Capability retention | `<= 3` pp drop on `word_2step` / IFEval / general retention sample |
| Structural invariant | Bit-exact identical parameter count (`873,438,784`), shapes, runtime graph, `153/153` frozen vision tensors, 0 gradient steps |
| Device efficiency | Peak memory (`~2.37 GB` MPS) and latency (`~109 tok/s`) on M4 Max within `10%` of `C0_raw` |

---

## 7. \$20.00 Pilot Gate and Go / No-Go Criteria

- **M1 Go-Criteria Status (ALL SATISFIED):**
  1. Reproducible baselines established on M4 Max MPS (`109 tok/s` `0.8B`, `64 tok/s` `2B`, zero NaNs).
  2. Donor `Qwen/Qwen3.5-2B` leads `Qwen/Qwen3.5-0.8B` by **+38.0 pp** on `carry_add_sub` (`40% -> 78%`), **+18.0 pp** on `mult_2x2_3x1` (`68% -> 86%`), and **+20.0 pp** on `gsm8k_val` (`62% -> 82%`).
  3. Remote HTTP `206` reader verified bit-exact at `9,893x` byte reduction.
- **M2 Authorized Pilot Scope (`D008`):** Stage 1 evaluates both `T1` (weights-only, paired with `C4_weights`) and `T2` (activation-assisted, paired with `C4_act`) strictly unpooled within the authorized \$20.00 pilot gate.

---

## 8. Milestone M3 Confirmatory Freeze Addendum (2026-11-26)

Per researcher approval recorded in [`docs/decisions.md`](decisions.md#d011--skip-stage-2-qwen35-9b-and-freeze-m3-confirmatory-scope) (`D011`; originally `D010` at freeze commit `d12e8fb`, renumbered in `docs/decisions.md` without altering any frozen protocol parameter), Stage 2 (`Qwen/Qwen3.5-9B`), Stage 3 (`google/gemma-4-E2B-it`), and Stage 4 (`moonshotai/Kimi-K3.1-Pro`) are skipped following verified negative Stage 1 validation results on both `T1` and `T2`. Before opening any file inside `/Users/kaiwendu/.awst_isolated_eval/`, the M3 confirmatory evaluation is frozen as follows:

- **Frozen 10 Arms:** `C0_raw`, `Qwen3.5-2B`, `C1`, `C3` (`alpha=0.05`, layers `8–23`, `seed=42`), `C4_weights`, `C4_act`, `T1`, `C5_T1_rowperm` (`seed=42`), `T2`, and `C5_T2_norm_matched` (`seed=42`). Single-session (`N=1` per track) replication scope is noted explicitly per Section 6 / Protocol Section 11.
- **Frozen Isolated Test Splits (`/Users/kaiwendu/.awst_isolated_eval/`):**
  - `gsm8k_test.jsonl`: `n=200` deterministically sampled via `sorted(random.Random(20261126).sample(range(1319), 200))`.
  - `synthetic_test.jsonl`: `n=150` balanced (`50` per sub-skill: `carry_add_sub`, `mult_2x2_3x1`, `word_2step`).
  - `synthetic_ood_test.jsonl`: full `n=100`.
- **Retention & Perplexity (`src/evaluation/retention.py`):** 30-item non-math IFEval-style instruction-following suite + 10-passage general-prose perplexity (`NLL` & `PPL`).
- **Holm-Bonferroni Primary Contrasts:** Adjusted separately within Track `T1` (`{T1 vs C1, T1 vs C4_weights}`) and Track `T2` (`{T2 vs C1, T2 vs C4_act}`), alongside `C5` donor-dependence, `H3` alignment (`C4`/`T1`/`T2` vs `C3`), 10,000-resample paired bootstrap 95% CIs, and M4 Max device efficiency checks. Raw per-item isolated test logs are stored strictly at `/Users/kaiwendu/.awst_isolated_eval/m3_logs/`.

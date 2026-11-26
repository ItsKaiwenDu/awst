# H003 · Activation-assisted arithmetic transfer

On the disjoint development follow-up, the half-strength T2 checkpoint solved **27/64 carry** and **38/64 GSM8K**. C0 solved 26/64 and 36/64. The archived activation-assisted C4 baseline solved 22/64 and 36/64. T2 gained one carry answer (+1.56 percentage points) and two GSM8K answers (+3.13 points) over C0. All four paired intervals against C0 and C4 include zero. The development evidence for improved arithmetic is inconclusive. All three checkpoints saw identical prompts and decoding settings. This pilot measures one deterministic checkpoint on development data. [Paired results](dev_results.json)

## Complete development trial log

Within each family, calibration used zero-based rows 0–15, the screen 16–35 and the follow-up 36–99. The three slices have disjoint IDs. Both T2 strengths came from the pristine recipient. On the screen, the half-strength edit won 23/40 to 19/40. We fixed it before opening the follow-up. We retained the full-strength checkpoint as the rejected candidate. [Trial starts](trial_starts.jsonl) · [Frozen choice](selection.json)

| Evaluation | Arm | Carry correct | GSM8K correct | Recipient forward calls | Generated tokens | Wall seconds |
|---|---|---:|---:|---:|---:|---:|
| [screen](c0_screen.json) | C0 raw | 6/20 | 13/20 | 17,886 | 17,849 | 163.27 |
| [screen](c4_screen.json) | C4 act | 1/20 | 12/20 | 17,191 | 17,156 | 157.72 |
| [screen](half_screen.json) | T2 0.5 | 7/20 | 16/20 | 19,315 | 19,280 | 179.12 |
| [screen](full_screen.json) | T2 1.0 | 6/20 | 13/20 | 19,114 | 19,079 | 173.91 |
| [followup](c0_followup.json) | C0 raw | 26/64 | 36/64 | 56,970 | 56,857 | 516.93 |
| [followup](c4_followup.json) | C4 act | 22/64 | 36/64 | 55,526 | 55,412 | 504.18 |
| [followup](half_followup.json) | T2 0.5 | 27/64 | 38/64 | 60,117 | 60,006 | 545.35 |

| Follow-up comparison | Family | Newly correct | Previously correct and lost | Difference (pp) | Paired 95% interval (pp) |
|---|---|---:|---:|---:|---:|
| T2 vs C0 | carry | 5 | 4 | +1.56 | [-7.81, +10.94] |
| T2 vs C0 | gsm8k | 3 | 1 | +3.12 | [-3.12, +9.38] |
| T2 vs C4 | carry | 10 | 5 | +7.81 | [-3.12, +18.75] |
| T2 vs C4 | gsm8k | 6 | 4 | +3.12 | [-6.25, +12.50] |

Intervals use 10,000 paired problem resamples and reflect sampling across these development prompts. Screen contrasts are selection-biased. The follow-up uses 64 examples per family and one editing session. [Scoring and intervals](summarize.py)

## What was localized

Before probing, we hypothesized that the donor would route numeric tokens more strongly through a few middle-to-late DeltaNet heads and that removing a matched donor head would hurt arithmetic readout more than removing its recipient counterpart. We also hypothesized that transplanting its memory decay would improve carry solutions while preserving GSM8K accuracy. We froze the procedure before the first forward at starting revision `cb8d828a125fe7f409bd3bdcf5409fba6e8879ec`. The localization and both editing strengths were committed in `939e4ee` before either T2 candidate completed its development screen. [Pre-edit design](design.md) · [Probe implementation](probe.py)

On 16 carry and 16 GSM8K development questions, each model received the normal chat prompt followed by the same assistant cue, `The final answer is \boxed{`. One prefill provided logits for the first numeric digit. We compared the gold digit with its strongest digit competitor. Fourteen carry and eleven GSM8K prompts had a higher digit margin in the donor and contributed to the selectivity contrast. This proxy uses the development gold digit for scoring. Whole-problem correctness was measured separately by uncued generation. [Per-prompt margins](localization.json)

Across 12 DeltaNet layers, we assigned 16 donor heads per layer to recipient heads using correlations of the write gate and log decay magnitude on paired question tokens. Head readout selectivity was the log ratio of RMS output on numeric versus other tokens, before the output projection. We ablated each shortlisted head at every token position on six carry and six GSM8K prompts. Positive margin drops indicate that ablation hurt the digit readout. [All 192 pair rankings and signed drops](localization.json) · [Per-forward ledger](probe_calls.jsonl)

| Layer | Donor → recipient head | Routing correlation | Digit selectivity gap, carry / GSM8K | Mean donor drop | Mean recipient drop | Donor minus recipient drop |
|---|---|---:|---:|---:|---:|---:|
| 14 | 5 → 5 | 0.7544 | +0.729 / +0.538 | −0.0417 | +0.0938 | −0.1354 |
| 13 | 12 → 12 | 0.9327 | +0.582 / +0.289 | −0.0104 | −0.0313 | +0.0208 |

The signed rule selected layer 13, head 12. Both models had zero mean ablation effect on the six carry probes for that head. On the six GSM8K probes, removal increased the donor margin by 0.0208 and the recipient margin by 0.0625. Neither candidate showed a positive mean donor margin drop across the 12 probes. The routing contrast supports the first hypothesis at the selected site. The ablations provide little evidence that the head was necessary for the donor's arithmetic.

## Transplant and controls

The source is the local `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`. The recipient is `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`. The probe verifies the pinned shard digests before loading either model. Both use the existing Qwen runtime with local files only. [Pinned shard digests](localization.json)

The two stored slots are index 12 of `model.language_model.layers.13.linear_attn.A_log` (float32) and `model.language_model.layers.13.linear_attn.dt_bias` (bfloat16). For each slot we used `cast(r + strength × (target − r))`. The target for `A_log` is the donor scalar. For `dt_bias`, it is the donor scalar plus the donor-minus-recipient difference in the median prompt-mean decay projection on numeric tokens. Those medians were −1.3248698711 and −1.3890380859, giving a +0.0641682148 offset. The remaining head weights and all other recipient tensors passed content checks. [Application code](apply_t2.py)

| Strength | Stored A_log | Stored dt_bias | Changed entries | Overall relative Frobenius norm |
|---|---:|---:|---:|---:|
| Pristine | −1.3359375 | −0.875 | 0 | 0 |
| 0.5 | −1.44140625 | −0.74609375 | 2 | 0.000176948988 |
| 1.0 | −1.546875 | −0.6171875 | 2 | 0.000353897976 |

At the median digit input, the scalar decay term implies a memory half-life of 26.66 token steps for C0, 26.22 at strength 0.5, 25.80 at 1.0, and 25.76 for the donor, using `log(2) / (exp(A_log) × softplus(a + dt_bias))`. This calculation isolates the decay term and holds its input constant. DeltaNet's update gate and actual token sequence also affect the recurrent state. [Half-strength provenance](half_patch.json) · [Full-strength provenance](full_patch.json)

The reference arms were a fresh run of pristine C0 and the existing T2 conventional baseline C4_act (APM plus ridge, alpha 0.02). The C4 checkpoint matched the archived manifest on changed tensor count, changed parameter count, edit norm and frozen vision checks. C4 changes 168,325,062 stored entries with relative Frobenius norm 0.004408710. Its historical calibration used 32 donor passes. The difference in edit scope and calibration budget limits mechanism comparisons. [C4 reference verification](c4_reference.json)

## Resource accounting

The T2 toolbox recorded **56/64 donor forwards** and 56 recipient probe forwards, all singleton prefills on development prompts. The models generated zero probe tokens. The measured input counts were 4,145 donor and 4,145 recipient probe tokens, including ablations. Calibration took 14.25 s. The two candidate builds took 10.37 s together. [Forward ledger](probe_calls.jsonl) · [Localization](localization.json)

Three of 12 candidate evaluations were used, plus four reference evaluations. All seven completed on the first attempt. Across 544 complete-solution attempts, the evaluator processed 38,774 input tokens, generated 245,639 tokens and counted 246,119 recipient forwards. The H003 recipient total including probing was **246,175**. Evaluation wall time was 2,240.48 s (37.3 min). The per-run manifests include loading time and generation diagnostics. The CPU-only replay and reversion check took 18.17 s and ran concurrently with part of the C0 follow-up. [Development totals](dev_results.json) · [Replay receipts](replay.json)

The local donor shard is 4,548,221,488 bytes. Probing loaded it twice, once for clean routing and once for ablations. The activation-assisted experiment therefore used a full local donor. Outside inference and integrity hashing, the explicit edit and QA reads comprised five pairs of 16-entry donor vectors, 96 bytes per pair (480 bytes total including the preliminary dtype inspection). The six full donor digest scans were integrity reads. Physical disk I/O depends on the operating-system cache. All runs used zero gradient steps and no downloads.

The completed H002 meter was $6.041536. At 2026-11-27T03:04:36.451Z H003 accounted for 192,785 uncached and 7,264,000 cached input tokens plus 38,514 output tokens, including 11,929 reasoning tokens. At the handoff rates, H003 shadow usage was **$5.988440** and combined usage was **$12.029976** of $12.00 at this snapshot. Direct API and cloud spend was $0.00. Subsequent chat and submission operations fall after the meter cutoff. [Token counters and rates](accounting.json)

## Replay and independent verification

The selected checkpoint is `hf_cache/edited/h003_half`. Reapplying the public function regenerated the full shard digest `ea5c744fe4d6627c2faefee13e2c2ad97c44f3ad012c7adc491d60290f6bb732`. Supplying the original donor vectors via `donor_override` produced the same bytes while the donor directory was inaccessible. A joint seeded permutation of the two vectors produced a different shard and the same count of two changed entries. Its edit norm was 0.0023574503, about 13.3 times the T2 norm. This checks source sensitivity. An independent behavioral C5 comparison should report that difference in perturbation size. [Replay, permutation and reversion receipts](replay.json)

Both candidate checkpoints retained 873,438,784 stored parameter entries, all 153 frozen vision tensors, the tokenizer files, 24 text layers and tied input/output embeddings. The pinned runtime instantiates 852,985,920 parameters in 517 modules for both pristine and edited checkpoints. Its loader omits the 15 serialized multi-token-prediction tensors. The module, parameter, buffer and alias signatures matched. Subtracting the finalist delta in float64 restored every stored value and the original shard's SHA-256. [Patch and runtime checks](half_patch.json) · [Exact reversion](replay.json)

From the repository root, the frozen application can be replayed into fresh paths with the project environment.

```sh
.venv/bin/python runs/codex_h003/apply_t2.py \
  --donor-dir hf_cache/models/qwen3.5_2b \
  --recipient-dir hf_cache/models/qwen3.5_0.8b \
  --out-dir hf_cache/edited/h003_replay \
  --manifest-path runs/logs/h003_replay_patch.tmp
```

For C5, call `apply_t2(donor_dir, recipient_dir, out_dir, manifest_path, donor_override=values)` from `runs.codex_h003.apply_t2`. Supply complete, 16-entry vectors for the two named tensors. Both keys are required. The ordinary and override paths use the same frozen head, activation offset, interpolation strength and storage rounding. This control tests dependence on the transplanted donor values conditional on the clean activation localization. Independent behavioral C5 scoring, blinded validation and any stage decision remain with Antigravity. [Public entry point](apply_t2.py) · [Executable override example](verify_replay.py)

The first fresh meter after the resumed chat showed $12.029976 combined shadow usage at 7:04:36 PM PST on November 26, $0.029976 above the shared $12 cap. The development runs and local verification had already finished. The final results are in the working tree. The final results commit and push are pending a budget decision. The published localization commit is `939e4ee`. The meter is a cutoff snapshot.

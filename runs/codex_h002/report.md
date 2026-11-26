# H002 · Arithmetic digit embeddings in the Stage 1 pilot

## Development outcomes

The weights-only transfer gained 22 correct GSM8K answers over the unedited recipient and lost five carry/borrow answers. Its GSM8K advantage also appeared on the 480 development items outside the selection screen, with 38 newly correct and 19 newly wrong answers. Neither edit improved carry/borrow over the base. These are development results from one agent session.

| Arm | All carry/borrow | All GSM8K | Carry outside screen | GSM8K outside screen |
|---|---:|---:|---:|---:|
| C0 raw | 110/300 (36.7%) | 264/500 (52.8%) | 103/280 | 254/480 |
| C1 donor-free, 1/32 | 103/300 (34.3%) | 267/500 (53.4%) | 94/280 | 255/480 |
| T1 weights-only, 1/32 | 105/300 (35.0%) | 286/500 (57.2%) | 99/280 | 273/480 |

Against C1 at the same edit radius, T1 gained 19 GSM8K answers and two carry answers on the full development set. Against C0 it fell short of the preregistration's numerical benchmark of +5 percentage points on GSM8K and positive carry gain. The confirmatory criteria apply to held-out evaluation. [Paired counts and per-problem correctness](dev_results.json)

## Design recorded before any generation or checkpoint edit

The researcher submitted H002 against main at `2dda64524e9bc324cb6aaca5cf6fdef84be30b2b`. The export at that revision and the submitted brief are byte-identical. The recipient tokenizer encodes the digits 0 through 9 individually at IDs 15 through 24. These ten rows serve as both input embeddings and output classifiers.

- **C1 hypothesis.** Confusion among plausible answer digits may arise when the recipient assigns too little contrast to distinct numerals. Expand each original digit row away from the recipient's ten-digit centroid. This increases between-digit separation while preserving the centroid in real arithmetic. It predicts fewer wrong digits, with a possible cost from overly confident intermediate steps. Freeze C1 selection before inspecting donor weights.
- **T1 hypothesis.** A donor may encode different relationships between written digits, number words and arithmetic operators. Transfer its *relative* digit-to-anchor similarities into the recipient's native basis. A regularized linear solve on recipient anchor rows supplies the direction. Only the ten digit rows are edited. This is a hypothesis about numerical representations in the tied vocabulary, with no assumed identification of an internal carry circuit.
- **Matched intervention.** Use the same ten rows and two relative edit radii, 1/32 and 1/8 of the Frobenius norm of the recipient's centered digit rows, in each arm. Compute on CPU in float64, then round once to the original bfloat16. C1 and T1 may have slightly different norms and changed-entry counts after rounding.
- **Screen and selection.** Follow the existing M2 development screen, the first 20 carry/borrow items and first 20 GSM8K items in file order. Select separately in each arm by total correct out of 40, breaking ties in favor of 1/32. Freeze one candidate even if both lose to the base. Apply the same base and the two selected checkpoints to the remaining 280 carry and 480 GSM8K items. Report the screen, untouched development remainder and full development totals separately. Count the finalist extension as an additional trial against each arm's cap. There is no further tuning on the remainder.
- **Evaluation.** Use the frozen repository prompt, parser, greedy decoding, thinking disabled, 1,024 new tokens, bfloat16, batch size one on M4 Max MPS. Use only the two named development files. The unit of paired comparison is the problem. Count prompt generations and actual top-level model forward calls separately.

## Adaptation formulas

Let `R` be the ten recipient digit rows, `Z = R − mean_rows(R)`, and `a` the stated radius. C1 writes `BF16(R + a Z)`. There is no donor handle in the C1 patcher or in the scoring process.

T1 reads the same ten digits and 30 arithmetic anchors from each embedding table. The anchors are the spelled-out digits, five arithmetic symbols and fifteen arithmetic words. The leading spaces in the word tokens matter. The exact strings, order and token IDs are fixed in `apply_t1.py`. For each model, center the thirty anchor rows and normalize each resulting row to unit length to obtain `A`. Form `C = Z Aᵀ / ‖Z‖F`. With subscripts `d` for donor and `r` for recipient, compute

```text
F = center_rows((C_d − C_r) (A_r A_rᵀ + I/8)⁻¹ A_r)
R′ = BF16(R + a ‖Z_r‖F F / ‖F‖F)
```

The code solves the 30-dimensional linear system directly. The cross-model comparison uses an identically indexed 10 × 30 table of similarities. The recipient and donor have different hidden widths. The common token identities supply the correspondence. The ridge constant and anchor list were fixed before donor access.

## Unedited recipient diagnostic

The first launch stopped at a pre-generation assertion. The frozen checkpoint has 488 stored tensors and 873,438,784 entries. The repository's Transformers class instantiates 473 unique parameters with 852,985,920 entries, leaving the 15 stored `mtp.*` tensors (20,452,864 entries) outside the active model. The corrected evaluator checks the measured runtime count, layer count, bfloat16 embeddings, tied weights and a runtime parameter schema hash. The patcher checks the full on-disk count, shapes, dtypes and frozen vision tensors.

On the 40-item development screen the base solved 7/20 carry and 10/20 GSM8K problems. All 40 responses yielded a numeric parse. Of 13 wrong carry answers, six had the correct units digit and one differed by an exact power of ten. Three carry responses looped, three reached the token limit and two did both. Two GSM8K responses reached the limit.

## C1 selection before donor inspection

The 1/32 edit scored 9/20 carry and 12/20 GSM8K. The 1/8 edit scored 8/20 carry and 13/20 GSM8K. Both total 21/40. The prespecified tie-break selected **1/32**. All C1 design, screening and selection were complete before opening the donor checkpoint. The independently scored C1 remainder uses this fixed checkpoint.

## Trial log and edit sizes

| Run | Carry | GSM8K | Changed entries | Overall relative Frobenius norm | Recipient forward calls | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| C0 screen | 7/20 | 10/20 | 0 | 0 | 18,040 | 168.42 |
| C1 screen 1/32, selected | 9/20 | 12/20 | 9,097 | 0.00003961378766 | 17,579 | 163.30 |
| C1 screen 1/8, discarded | 8/20 | 13/20 | 9,947 | 0.00015738576884 | 17,750 | 165.21 |
| T1 screen 1/32, selected | 6/20 | 13/20 | 9,050 | 0.00003952586910 | 16,882 | 155.28 |
| T1 screen 1/8, discarded | 7/20 | 10/20 | 9,960 | 0.00015745551641 | 17,207 | 158.54 |
| C0 remainder | 103/280 | 254/480 | 0 | 0 | 332,151 | 3,060.92 |
| C1 remainder | 94/280 | 255/480 | 9,097 | 0.00003961378766 | 336,692 | 3,096.85 |
| T1 remainder | 99/280 | 273/480 | 9,050 | 0.00003952586910 | 332,564 | 3,029.24 |

For T1, the donor-to-recipient similarity error was 0.18545 before editing. It fell to 0.15158 at 1/32 and 0.05412 at 1/8. The larger reduction in this representation metric produced the lower development screening score. Discarded checkpoints were removed after scores and patch manifests were saved. [T1 selected patch](t1_low_patch.json) · [T1 discarded patch](t1_high_patch.json) · [C1 selected patch](c1_low_patch.json) · [C1 discarded patch](c1_high_patch.json)

Across all 800 development items, C0, C1 and T1 respectively had 37, 40 and 43 carry responses reach the token cap, and 43, 47 and 35 GSM8K responses reach it. Each finalist produced a parseable numeric answer for every problem. Under this decoding setup, the observed GSM8K difference includes changes in completion behavior.

## Resource and integrity checks

| Scored checkpoint group | Prompt generations | Actual recipient forward calls | Generated tokens | Evaluation seconds | Charged candidate trials |
|---|---:|---:|---:|---:|---:|
| Shared C0 | 800 | 350,191 | 349,471 | 3,229.34 | 0 |
| C1 | 840 | 372,021 | 371,273 | 3,425.35 | 3/12 |
| T1 | 840 | 366,653 | 365,894 | 3,343.06 | 3/12 |
| All | 2,480 | 1,088,865 | 1,086,638 | 9,997.74 | 6 |

The forward calls include prefill and autoregressive decoding, measured with a top-level model hook. Each arm used two screening trials and one finalist extension. The elapsed scoring time was 2 h 46 min 38 s. C1 had zero donor accesses. T1 used 163,840 donor weight bytes per patch across two candidates and one default-script replay, for **491,520 weight bytes**. Its separately metered tokenizer and cache-receipt check read 12,808,107 donor metadata bytes. The combined charge was **13,299,627 bytes** against 200,000,000 allowed. These figures use toolbox payload accounting. Both arms used zero donor forward passes and zero gradient steps.

All stored shapes and dtypes matched. Each finalist retained 873,438,784 stored parameters, all 153 frozen vision tensors, the configuration, tokenizer and chat template. Reloads of the base and both finalists had identical 517-module trees, runtime parameter and buffer schemas, and tied embedding aliases. The default CLI recreated each entire finalist shard byte for byte. The pristine recipient shard also matched its pinned hash after both replays. The retained checkpoints are `hf_cache/edited/h002_c1_low` and `hf_cache/edited/h002_t1_low`. [Replay hashes and exact edit norms](replay_check.json)

At 1:48:07 PM PST on November 26, the two H002 chat turns used 5,223,861 input tokens (5,055,488 cached) and 43,379 output tokens (18,196 reasoning). The measured shadow cost was $5.1043392 at the handoff rates, with $0.00 direct API or cloud spend. The meter excludes later delivery tokens. [Detailed accounting](accounting.json)

## Checkpoint provenance and reproduction

The recipient is `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`. Its pristine shard SHA-256 is `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The donor is `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`. The cached shard receipt matches its pinned revision and ETag. The forty donor rows, in manifest order, have raw bfloat16 SHA-256 `997caa141917f0db83917d6e23ef136f910f8a6b989499bfaf9e84a844b2ec37`. Each row comes from `model.language_model.embed_tokens.weight`. Hashing and decoding both tokenizers verified the same forty IDs and byte-identical tokenizer files. [Source audit](source_audit.json)

From the project root, the commands below recreate the selected checkpoints in fresh output directories. Add `--strength low` or `--strength high` to reproduce a particular screen candidate.

```sh
.venv/bin/python runs/codex_h002/apply_c1.py --recipient-dir hf_cache/models/qwen3.5_0.8b --out-dir hf_cache/edited/h002_recreated_c1 --manifest-path hf_cache/edited/h002_recreated_c1.json
.venv/bin/python runs/codex_h002/apply_t1.py --donor-dir hf_cache/models/qwen3.5_2b --recipient-dir hf_cache/models/qwen3.5_0.8b --out-dir hf_cache/edited/h002_recreated_t1 --manifest-path hf_cache/edited/h002_recreated_t1.json
```

For independent C5, the repository's `build_shuffled_donor_override(donor_dir, target_layers=[], include_embed=True, seed=42)` supplies the embedding tensor. Pass the resulting dictionary as the fifth argument to `apply_t1`. A test supplies an inaccessible donor path and checks that identical override values reproduce the ordinary formula bit for bit, while another source changes the result. This checks the software contract. Antigravity owns the scored C5, float64 reversion and blinded validation.

# Literature Review and Novelty Reassessment (Milestone M0)

**Date:** 2026-11-25  
**Scope:** Primary protocol references (`AWST_Research_Protocol_v0.3.md` Section 15) plus newly published November 2026 Weight-Space Programming (WSP) and cross-scale merging work.  
**Verification note:** All entries below reflect metadata, abstracts, and technical summaries verified on 2026-11-25. Full PDF inspection of mathematical formulations and hyperparameters is marked `[verify full text before submission]` where needed before baseline implementation (`M2`) or manuscript drafting (`M4`).

---

## 1. Detailed Paper Reviews

### 1.1 Sparks of Weight-Space Programming (`takehi2026sparks`)
- **Citation:** Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026, Nov 2). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL blog + public logs (`RikiyaT/lm_understand-logs`).
- **Method & Findings:** Demonstrates that frontier LLMs (specifically GPT-6 Astra) can inspect raw neural network weights, write small MinAtar/Atari policy networks from scratch, and perform surgical edits on open-weight LLMs via a Python kernel and forward-pass inspection on practice problems. On `Qwen3.5-0.8B`, the agent diagnosed repetitive reasoning loops caused by a specific vector direction in MLP layers 15 and 19. Subtracting this direction modified **131,072 weights (~0.016% of parameters)** and increased:
  - **GSM8K:** 13.8% -> 67.1%
  - **MATH-500:** 7.9% -> 48.1%
  - **IFEval:** 21.0% -> 62.3%
- **Limitations & Gaps:**
  - The dramatic math gain on `Qwen3.5-0.8B` was primarily **loop suppression** (unblocking latent reasoning), not transplantation of new mathematical circuits.
  - Their Gemma experiment did not establish acquisition of new mathematical ability.
  - They explicitly proposed cross-model circuit grafting as future work without testing it.
- **Direct Implication for AWST:** **Critical confound.** If AWST runs on raw `Qwen3.5-0.8B` (`C0_raw`), any edit that suppresses reasoning loops can yield massive accuracy jumps even with zero genuine math transfer from the donor. Stage 0 must replicate/characterize this loop-suppression baseline (`C0_loopfixed`), and Stage 1 must separate loop repair (`C1`) from donor-dependent mathematical transfer (`T1`/`T2`). See `docs/decisions.md` (`D002`).

### 1.2 Training-Free Knowledge Transfer Across Model Scales through Activation-Guided Pruning (`fan2026apm`)
- **Citation:** Fan, J., Chen, S., Hou, Y., Zhang, A., & Xie, H. (2026, Aug). arXiv:2608.13596.
- **Method & Findings:** Proposes **Activation-Prune-Merge (APM)**, a non-agentic, training-free framework for heterogeneous cross-scale transfer from a larger donor to a smaller recipient without explicit neuron-wise semantic alignment. APM: (1) runs the donor on task data to construct task-conditioned activation maps, (2) prunes salient layers, hidden dimensions, attention heads, and MLP neurons down to recipient dimensions, and (3) injects the compressed slice via micro-interpolation. Across 16 benchmarks, APM raised average accuracy of a 3B recipient from 55.5% to 60.6% without gradient training. `[verify full text before submission]`
- **Limitations & Gaps:** Requires running the donor model on calibration data (incompatible with a strict weights-only track) and relies on algorithmic pruning + micro-interpolation rather than agentic reasoning about functional modules.
- **Direct Implication for AWST:** Proves that **cross-scale training-free transfer already exists algorithmically**. Cross-scale transfer alone cannot be claimed as AWST's novelty. APM is a mandatory **`C4` conventional baseline** for the activation-assisted track (`T2`), and inspires a weights-only alignment analog for `C4` under `T1`.

### 1.3 Transport and Merge: Cross-Architecture Merging for Large Language Models (`cui2026transport`)
- **Citation:** Cui, C., Yang, B., Shen, F., Chen, Y., Zheng, J., & Wang, X. (2026, Feb). arXiv:2602.05495.
- **Method & Findings:** Uses **Optimal Transport (OT)** over calibration activations from both source and target LLMs to compute functional neuron correspondences across heterogeneous architectures, then merges weights into the target without changing its architecture. Improves low-resource language and specialized domain performance with minimal calibration data. `[verify full text before submission]`
- **Limitations & Gaps:** Requires activation samples from both donor and recipient; depends on OT solver assumptions and calibration set coverage.
- **Direct Implication for AWST:** Directly reinforces `H3` (alignment-aware transfer beats unaligned slice copying) and provides a principled closed-form alignment baseline (`C4`) for Stage 3 (cross-family `Qwen3.5-9B -> Gemma 4 E2B`) and Stage 1 (`2B -> 0.8B`) activation-assisted comparisons.

### 1.4 WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming (`li2026weightcode`)
- **Citation:** Li, B., Cui, T., Zhang, B., Li, M., Zhang, C., Zhang, Y., Heng, P. A., Ouyang, W., Yin, Z., Wu, Y., & Yang, L. (2026, Nov). arXiv:2611.25476.
- **Method & Findings:** Uses a frozen LLM to generate **executable weight programs** that inspect task specifications (rules, demonstrations) alongside existing model weights to compute closed-form parameter updates. Reports strong retention and outperforms fine-tuning on capability-installation and ImageNet/robustness checks, and shows weight programs can transfer across independently trained models of the *same* architecture. `[verify full text before submission]`
- **Limitations & Gaps:** Synthesizes updates from task rules/examples rather than extracting and adapting learned circuits from a larger, differently sized donor LLM; cross-seed transfer was demonstrated within a shared architecture, not across heterogeneous scales/families.
- **Direct Implication for AWST:** Validates the paradigm of LLMs writing reusable weight-manipulation code (rather than editing raw floats manually in context), while leaving **donor-to-recipient cross-scale weight transplantation** untested.

### 1.5 Replicating and Extending Weight-Space Programming (`goworek2026replicating`)
- **Citation:** Goworek, R., Hickey, A., Nanni, F., & O'Reilly, M. (2026, Nov). The Alan Turing Institute. arXiv:2611.21449.
- **Method & Findings:** Independently replicates Takehi et al. (`takehi2026sparks`) on writing policy networks (e.g., *Breakout*) and decompiling weights back to Python, but tests reading ability on custom-designed tasks. Crucially finds that frontier LLMs' weight-reading success relies heavily on **decoding input-layer configurations** rather than genuinely comprehending deep internal algorithmic mechanics, and that shallow weight auditing struggles to isolate exact triggers in poisoned LLMs. `[verify full text before submission]`
- **Limitations & Gaps:** Evaluates single-model reading/writing rather than donor-to-recipient transfer.
- **Direct Implication for AWST:** **Direct mechanistic risk for Track `T1` (weights-only).** If frontier LLMs mostly read input-layer cues rather than deep internal circuits, weights-only `T1` may fail to identify genuine mathematical sub-circuits in a 2B/9B donor without activation probes (`T2`) or numerical alignment tools. This makes the `T1` vs `T2` vs `C4` vs `C5` comparison scientifically pivotal regardless of sign.

### 1.6 Git Re-Basin (`ainsworth2023gitrebasin`) and ZipIt! (`stoica2024zipit`)
- **Citations:** Ainsworth, Hayase, & Srinivasa (ICLR 2023, arXiv:2209.04836); Stoica et al. (ICLR 2024).
- **Findings & Relevance:** Independently trained networks exhibit permutation symmetries and basis rotations; naive elementwise weight copying or averaging degrades performance due to representation misalignment. `Git Re-Basin` solves permutation matching across identical architectures; `ZipIt!` merges models trained on disjoint tasks by zipping redundant features within/across models. `[verify full text before submission]`
- **Direct Implication for AWST:** Motivates `H3` (alignment-aware adaptation is required even between `Qwen3.5-2B` and `Qwen3.5-0.8B`) and explains why direct tensor-slice copying (`C3` or naive `T1`) should fail unless representation bases are aligned.

### 1.7 Newly Identified Late-2026 WSP Extensions (`ropinski2026editready`, `gross2026wspverify`)
- **`ropinski2026editready` (arXiv:2611.26408, Nov 2026):** *Training Vision Models for LLM-Driven Weight-Space Programming* (Ropinski & Kranz). Shows standard ERM-trained vision models are structurally rigid under direct LLM weight edits, motivating *Edit-Ready Training (ERT)* during pretraining/training to make weights cooperative with GPT-6 Astra / Claude Opus edits. **Implication:** Standard pretrained open-weight LLMs (`Qwen3.5`, `Gemma 4`) were *not* trained for edit-readiness, increasing the prior probability of negative or fragile results in AWST unless closed-form alignment (`H3`) compensates.
- **`gross2026wspverify` (arXiv:2611.11604, Nov 2026):** *Weight-Space Programming of Policies and Their Verification via Model Checking* (Gross). Integrates formal verification (Storm model checker) into WSP loops to catch unsafe/broken policy edits. **Implication:** Reinforces AWST's strict pre-evaluation integrity checks (architecture invariants, retention checks on IFEval/general text, revert-on-regression acceptance rules).

*(Note: M0 search also surfaced recent November 2026 discussion of `WSP-Scope` benchmarks and critiques asking whether weight reading reflects structural reasoning vs memorization of public distributions; verify primary arXiv IDs/authors before citing in `M4`.)*

---

## 2. Reassessed Novelty Claim and Key Experimental Safeguards

Following M0 literature verification, the novelty and risk profile of Project AWST is sharpened as follows:

1. **What is NOT novel:**
   - Agent-driven single-model repair (`takehi2026sparks`, `li2026weightcode`).
   - Non-agentic training-free cross-scale or cross-architecture transfer (`fan2026apm`, `cui2026transport`).
2. **What IS novel and falsifiable in AWST:**
   - A **controlled, preregistered empirical test of donor-dependent, agent-directed weight transplantation** between distinct pretrained language models (`Qwen3.5-2B/9B -> Qwen3.5-0.8B`, and optionally `Qwen3.5-9B -> Gemma 4 E2B`), rigorously separating:
     - **Loop repair / generic recipient WSP (`C1`)** from genuine **donor-dependent math capability transfer (`T1`/`T2` > `C1` and `C5`)**;
     - **Agent-directed selection & adaptation (`T1`/`T2`)** from **conventional algorithmic donor transfer (`C4`)**;
     - **Weights-only inspection (`T1`)** from **activation-assisted transfer (`T2`)** under the known shallow-reading limitation (`goworek2026replicating`).
3. **Critical design adjustments adopted for M1/M2 (see `docs/decisions.md`):**
   - Evaluate Stage 1 against **both** raw `Qwen3.5-0.8B` (`C0_raw`) and loop-suppressed `Qwen3.5-0.8B` (`C0_loopfixed`) so loop repair cannot masquerade as donor math transfer (`D002`).
   - Freeze `Qwen3.5` multimodal vision towers and explicitly account for the hybrid `3:1` Gated DeltaNet + Full Attention layout (`D003`).
   - Treat negative or null results (e.g., `T1` failing to beat `C1` or `C4`) as first-class publishable feasibility findings.

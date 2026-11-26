# Agentic Weight-Space Transplantation (AWST)

## Research proposal, theoretical checklist, and agent implementation brief

**Working paper title:** Agentic Weight-Space Transplantation: Training-Free Transfer of Mathematical Reasoning Between Language Models  
**Researcher:** [Name / affiliation]  
**Version:** 0.3 | November 25, 2026  
**Status:** Pre-experimental; all proposed results, thresholds and timelines are provisional  
**Primary domain:** Grade-school mathematical reasoning (arithmetic and word problems)  
**Secondary domain:** Coding (future extension)  
**Estimated duration:** 8 weeks for the initial study (10–15 hours/week); 3–5 months for an expanded study, subject to results.  
**Initial study spending ceiling:** $200 USD, with a $20 pilot authorization gate. No paid work until the researcher explicitly authorizes the pilot.

### Agreed project decisions (v0.3)

- Target grade-school mathematics, using a narrow arithmetic skill for early diagnostics and GSM8K as the primary confirmatory benchmark.
- Primary project harness: Google Antigravity, using Gemini 4 Neon or Carbon as available and approved. GPT-6.2 Astra in Codex is a secondary specialist, invoked through a researcher-mediated prompt handoff. Compare permitted agents on small tasks; do not assume general benchmark rankings predict weight-space competence.
- After each completed major milestone, update persistent research records, verify the changes, use `/git-commit` if available, commit with a short descriptive title, and push to the approved remote before continuing. Pause for human intervention whenever required.
- Start with two small-model transfer experiments: Qwen3.5-2B to Qwen3.5-0.8B, then Qwen3.5-9B to the same recipient if justified. These are candidate checkpoints pending compatibility, availability, license and baseline verification.
- Use local computing first. Stay under $200 for the initial study; estimated spending is $60–140, conditional on successful local execution and bounded agent usage.
- Kimi K3.1 Pro to Gemma 4 E2B is a gated stretch goal, not a commitment within the $200 budget. No full Kimi download or inference by default.
- The exact initial arithmetic subskill, permitted agent, final checkpoint revisions and confirmatory sample sizes remain open until pilot design and approval.

---

## 1. Working abstract

Existing methods for transferring knowledge between large and small language models often rely on knowledge distillation, fine-tuning, or mathematical weight-merging techniques. Recent research in Weight-Space Programming suggests that frontier language models can directly inspect and modify neural network weights without gradient-based training. However, whether AI agents can reliably select and transfer learned capabilities between independently trained language models remains an open question.

This study proposes Agentic Weight-Space Transplantation (AWST), a method in which an AI agent identifies, extracts, adapts, and transplants selected weights from a larger donor model into a smaller recipient model. The initial experiments focus on grade-school mathematics, including arithmetic and word problems, while preserving the recipient's architecture and parameter count. AWST will be evaluated against unmodified models, random weight transplantation, conventional weight-transfer algorithms, and donor-free Weight-Space Programming. Evaluation will measure mathematical accuracy, generalization, capability retention, computational cost, and inference efficiency.

The objective is to determine whether AI-directed weight transplantation can provide a practical alternative to conventional knowledge-transfer methods, particularly for resource-constrained language models. No empirical results are available at this stage.

**Keywords:** weight-space programming, model merging, knowledge transfer, mechanistic interpretability, language models, mathematical reasoning, on-device AI.

## 2. Research questions and hypotheses

**Primary question (RQ1):** Can AI agents selectively identify, extract, adapt, and transplant grade-school mathematical reasoning capabilities from larger pretrained language models into smaller language models without gradient-based training or increasing their parameter count?

**Secondary questions:**

- RQ2: Does AI-directed transplantation outperform conventional algorithm-directed transplantation under matched resource constraints?
- RQ3: Does access to a stronger donor provide additional benefit beyond AI-directed editing of the recipient alone?
- RQ4: Can the method generalize across model architectures and independent training runs?
- RQ5: Can weights-only remote access support useful transplantation without loading the entire donor model?
- RQ6 (exploratory): Which permitted agent performs best per unit of cost and compute?

**Primary hypothesis (H1):** AI-directed transplantation produces greater held-out mathematical accuracy gains than both donor-free AI editing and a conventional donor-based transfer method, while preserving the recipient architecture and parameter count.

**Null hypothesis (H0):** AI-directed transplantation has no reliable advantage over the matched controls. For the primary analysis, the null is that the expected gain over either key control is zero or negative.

**Secondary hypotheses:**

- H2: Improvement persists on previously unseen problems and is not explained solely by reduced repetition, truncation, or answer-format changes.
- H3: Alignment-aware transfer is more effective than direct copying of unaligned tensor slices.
- H4: A method using only selected remote donor tensors can achieve useful gains with substantially less donor data than a full checkpoint download.

These are hypotheses, not established properties of frontier models. Negative results are valid research outcomes.

## 3. Prior work and novelty

| Work | Relevance | Implication for AWST |
|---|---|---|
| Sparks of Weight-Space Programming (Takehi et al., 2026) [1] | AI-directed direct weight edits; proposes transferring circuits across models | Establishes motivation, not proof of the proposed transfer method |
| Activation-Prune-Merge (Fan et al., 2026) [2] | Training-free large-to-small transfer guided by donor activations | Strong conventional baseline; cross-scale transfer alone is not novel |
| Transport and Merge (Cui et al., 2026) [3] | Cross-architecture weight fusion using activation alignment | Relevant alignment method and baseline |
| WeightCode (Li et al., 2026) [4] | LLM-generated reusable weight programs; fixed edits transfer poorly | Shows why recipient-specific adaptation matters |
| Git Re-Basin (Ainsworth et al.) [5] | Aligns independently trained networks through permutations | Demonstrates why matching tensor indices can be misleading |
| Replicating and Extending Weight-Space Programming (2026) [6] | Independent tests of weight reading and programming | Useful for choosing a preliminary agent competency test |
| ZipIt! (2024) [7] | Training-free merging of independently trained networks | Additional related work and evidence of feature alignment challenges |

**Novelty claim (provisional):** The contribution is a controlled empirical test of agent-directed, donor-dependent transfer of a specific learned capability between fixed-size language models. The goal is to separate genuine donor contributions from generic editing, conventional transfer and random perturbations. Cross-family and remote-only transfer are extensions, not prerequisites for the initial study.

Do not claim to be the first study without an updated literature review. The original WSP authors explicitly proposed circuit grafting as future work. Their reported Qwen math improvement largely involved suppressing reasoning loops; their Gemma experiment did not establish acquisition of new math ability [1]. This motivates the more demanding AWST controls.

## 4. Operational definitions and scope

- **Donor:** A pretrained open-weight model with stronger measured performance in the target mathematical skill.
- **Recipient:** A smaller pretrained model whose existing parameter slots may be modified.
- **Agent:** A permitted AI model that makes and explains decisions about inspecting, selecting and adapting donor weights. It is distinct from the donor and recipient.
- **Transplantation:** A documented numerical transformation that depends on actual donor weight values and writes the result into existing recipient parameter slots.
- **Training-free:** No backpropagation, gradient descent, loss-driven parameter fitting, fine-tuning, distillation, or learned adapters in the primary AWST method. Forward-only measurements, closed-form alignment, bounded hypothesis-driven edits and evaluation are permitted. Record repeated evaluation-driven selection explicitly; do not describe it as zero optimization or zero data.
- **Fixed architecture:** Same parameter shapes, number of layers, tokenizer, runtime operations and parameter count. Do not add adapters, experts, external tools, or inference-time routing to the recipient.
- **Capability transfer:** An observed, reproducible donor-dependent improvement on held-out tasks, supported by controls. A verified transplanted functional circuit is a stronger claim requiring causal intervention evidence.
- **Weights-only track:** The agent can read donor weight metadata, numerical summaries and selected tensor slices, but cannot run the donor. Recipient forward passes are still allowed.
- **Activation-assisted track:** A separate experiment with access to donor activations as well as weights. Do not pool results with the weights-only track.

Out of scope: general intelligence improvement, self-improving agents, removing safeguards, production deployment, exhaustive trillion-parameter downloads, and coding benchmarks until the mathematics study is complete.

## 5. Theoretical checklist

### A. Feasibility of a learned capability

- [ ] Choose a narrow, measurable grade-school skill first: multi-digit arithmetic with carries, multiplication, or fraction operations. Multi-digit addition/subtraction with carries and borrowing is the proposed starting diagnostic; verify that it is neither at ceiling nor at floor before freezing the skill. GSM8K remains the primary confirmatory benchmark.
- [ ] Confirm the donor actually outperforms the recipient on development examples under a matched evaluation protocol.
- [ ] Check that recipient failures are genuine wrong answers, not primarily repetition, formatting or token limits.
- [ ] Establish whether the skill is plausibly represented by a localized or distributed set of features; do not assume a single math neuron, layer or expert exists.
- [ ] Consider whether the smaller recipient has enough representational capacity to use the donor feature.
- [ ] Separate recovery of latent ability, transfer of a new skill, and general behavior changes.

### B. Weight-space compatibility

- [ ] Inventory model configurations: layer count, hidden sizes, MLP sizes, attention heads, KV heads, normalization, activation functions, residual paths, positional encodings, tokenizer and tied embeddings.
- [ ] Map corresponding operations by function and data flow, not names or layer index alone.
- [ ] Account for independent training, neuron permutations, sign/scale conventions where applicable, and different representation bases.
- [ ] Check that a proposed donor component has compatible input/output semantics and tensor shapes after adaptation.
- [ ] Consider upstream and downstream dependencies: a single weight slice may not be functional in isolation.
- [ ] Treat dimensionality reduction as lossy; a donor circuit may require more rank or width than the recipient can represent.
- [ ] Compare attention and MLP dependencies separately. If gated MLPs are involved, account for paired input and output projections and gating conventions.
- [ ] For mixture-of-experts donors, distinguish shared layers, routed experts and routers; routing behavior is not known from expert weights alone.
- [ ] Account for tensor layout, transpose conventions, quantization scales, packed values and dequantization error.
- [ ] Check that unchanged tensors and tied parameters remain consistent after saving and reloading.

### C. Causal evidence and competing explanations

- [ ] Document the agent's proposed function of the selected donor component before editing.
- [ ] Record provenance of every donor tensor, transformation and recipient location.
- [ ] Compare with donor-free WSP at equal editing and evaluation budgets.
- [ ] Compare with random and norm-matched edits, and a control using shuffled or unrelated donor values.
- [ ] Where donor execution is available, test whether intervening on the proposed component changes the target skill more than matched alternatives.
- [ ] Revert transplanted weights to test whether the gain disappears; run matched perturbations to rule out nonspecific effects.
- [ ] Where feasible, test restoration or re-insertion and independent donor checkpoints.
- [ ] Assess transfer across unseen problem templates and difficulty levels.
- [ ] If donor execution is unavailable, describe results as donor-dependent weight transfer, not proven circuit transplantation.

### D. Research integrity

- [ ] Record exact model revisions, licenses, policy permissions, software versions, seeds and hardware.
- [ ] Freeze development/validation/test splits and primary metrics before final runs.
- [ ] Prevent the agent from reading hidden test questions, answers, results or benchmark-derived optimization signals.
- [ ] Preregister selection rules, resource limits, control methods and statistical analysis.
- [ ] Treat failed attempts as data and record all trials, not just best cases.
- [ ] Report known limitations: benchmark contamination, noisy inference, multiple comparisons, compute asymmetry and uncertain circuit localization.
- [ ] Revisit literature before submission and avoid unsupported priority claims.

## 6. Model selection and staged experiments

| Stage | Donor | Recipient | Goal |
|---|---|---|---|
| 0 | None | Qwen3.5-0.8B | Reproduce the published WSP baseline and diagnose looping |
| 1 | Qwen3.5-2B | Qwen3.5-0.8B | First small same-family donor-dependent experiment |
| 2 | Qwen3.5-9B | Qwen3.5-0.8B | Test a stronger donor with the same recipient and evaluate robustness |
| 3 | Qwen3.5-9B | Gemma 4 E2B | Cross-family transfer feasibility |
| 4 (optional) | Kimi K3.1 Pro | Gemma 4 E2B | Selective remote transfer from a very large donor |

Candidate model cards: [8–11, 16]. Same-family checkpoints may still differ substantially in architecture or representation basis. Verify exact compatibility, license terms, checkpoint format and actual parameter count rather than treating model names as specifications. Keep Stage 1 small; choose a different compatible pair if its baseline has too much looping or no useful performance gap. An independently trained recipient with limited looping is an optional diagnostic if loop repair dominates. Stage 4 is not required for a successful project and is outside the initial spending authorization.

## 7. Agent selection and permission gate

Candidate programmers: GPT-6.2 Astra (Codex), Gemini 4 Neon or Carbon (Antigravity), the original GPT-6 Astra for replication if available and approved, and a suitably licensed open-weight agent as fallback. General intelligence benchmarks are screening information, not a measure of weight-space understanding.

Before any provider-hosted agent is used for actual model improvement, review its applicable product/API agreement and obtain clarification if the intended use is uncertain. OpenAI and Google have restrictions involving competing models; open-to-open transfer and the absence of distillation do not automatically exempt the work [13–15]. Do not rely on another research team's access as permission. If uncertain, use an agent with an appropriate license on an authorized local or hosted runtime. Codex can be given this protocol for review and permitted preliminary work; do not assume permission to have it generate or execute donor-to-recipient improvement code.

**Pilot agent selection:**

- Prioritize a small matched comparison of Astra and Neon, with Carbon if affordable. If the budget or permissions prevent comparison, select one permitted agent and report that limitation. Before any paid calls, verify live pricing, including reasoning tokens, context-tier charges, caching and tool fees.
- Reproduce a small known WSP task, test interpretation of a synthetic network with a known function, and test whether agents identify a known corruption or alignment mismatch.
- Use matched tools, prompts, trial counts, forward-pass and token budgets; record exact provider/model version and reasoning setting.
- Evaluate accuracy of proposed mechanisms, reproducibility, resource use and failure rate. Select the agent using pilot data before final testing.
- Treat a scripted algorithm as a baseline, not an inferior method by assumption. Agents can use numerical algorithms as tools; their experimental distinction is control over the selection and adaptation decisions.
- Use local numerical tools to compute statistics, examine limited tensor samples and execute authorized operations. Never serialize entire weight matrices into the agent context. Limit tool output, context growth and API retries; record usage and enforce spending ceilings.

## 8. Experimental design and controls

| ID | Condition | Purpose |
|---|---|---|
| C0 | Original recipient | Establish baseline |
| C1 | AI editing recipient only | Separate ordinary WSP from donor transfer |
| C2 | Random recipient perturbations, matched edit norm and count | Test nonspecific effects |
| C3 | Random compatible donor slices, matched transfer budget | Test value of informed donor selection |
| C4 | Conventional donor transfer (e.g., activation-guided or alignment-based as applicable) | Test advantage over algorithm-directed transfer |
| C5 | AWST with shuffled/unrelated donor values, matched scale and shape | Test dependence on meaningful donor information |
| T1 | Agent-directed donor selection and adaptation | Primary AWST condition |
| T2 | Agent-directed donor transfer with donor activations | Optional extension; compare separately |
| C6 | Fine-tuning or distillation | Optional reference, separate compute/data regime; not part of training-free AWST |

Use the same base checkpoint, evaluation harness, permitted practice data, generation settings and reporting rules. For editable arms, match or explicitly report edit count, relative edit norm, number of candidate trials, forward passes, generated tokens, donor bytes accessed and time. If a conventional method requires donor activations, compare it against the activation-assisted agent track; use a weights-only algorithm in the weights-only comparison. Report unmatched budgets as limitations.

**Candidate edit protocol (conceptual):** Identify a target failure, inspect recipient and donor evidence, propose a bounded compatible change, record the hypothesis, apply to a temporary checkpoint, test on development data, accept or revert using frozen rules, then select one final candidate per run for hidden evaluation. No hidden benchmark feedback may reach the agent. Keep a pristine checkpoint and immutable patch log. Do not assume that copying weights alone preserves function.

## 9. Datasets and evaluation

| Dataset | Role |
|---|---|
| Seeded synthetic arithmetic problems | Narrow-skill pilot and secondary held-out diagnostic; controlled difficulty and novel templates |
| GSM8K | Primary confirmatory benchmark for grade-school word-problem reasoning |
| MATH-500 | Optional broader mathematical reasoning benchmark, if budget allows |
| IFEval or similar | Detect instruction-following regression |
| Small general-language/QA sample and held-out text | Detect general degradation and optionally measure perplexity |

**Split design:** For the synthetic dataset, use a fixed disjoint development/validation/test split (e.g., 60/20/20) with separate seeds and held-out templates. Use only official training material or independently generated development problems when working with GSM8K/MATH. Keep official benchmark test examples and answers in a separate evaluator. Public benchmarks may be in model pretraining; independently generated held-out tasks help address contamination but cannot eliminate it entirely.

**Primary endpoint:** Absolute percentage-point difference in exact-answer pass@1 on the frozen GSM8K test set between T1 and each of C1 and C4. The unmodified recipient (C0) is the reference for absolute improvement. The narrow synthetic skill test is an independent secondary endpoint and a pilot diagnostic. Secondary endpoints also include OOD synthetic accuracy, optional MATH-500, answer completion, repetition rate, truncation, invalid answer rate, general capability retention, latency, peak memory, output length and total costs. Freeze the exact test size and selection rule before confirmatory experiments; if the full GSM8K test is unaffordable, preregister a reproducibly sampled subset and disclose it. Do not use test feedback to select edits.

- Score incorrect, missing and truncated answers consistently; report them separately as well.
- Use identical prompt templates within each recipient, decoding settings, max generated tokens, stopping rules and seeds. Report any donor-specific template differences.
- Run a matched-token-budget diagnostic and a diagnostic among completed responses; the latter is secondary because completion can change the sample composition.
- Use deterministic answer checking where possible. Log parsing failures and manually audit a blinded sample. Report pass@1, not best-of-N as the primary metric.
- Repeat independent agent sessions, not only generation seeds. Aim for at least three pilot runs and five confirmatory runs per key arm if feasible; determine final sample size from pilot variability and a power analysis. Cluster repeated evaluation outcomes by run and problem, as applicable.
- Report paired effect sizes and 95% confidence intervals using an appropriate paired or hierarchical bootstrap; for a single deterministic paired checkpoint comparison, McNemar's test can be a supplementary analysis. Correct multiple primary comparisons (e.g., Holm); mark additional analyses exploratory. A few runs may leave uncertainty even with many problems.
- Freeze selection and analysis before opening test results. If a new method is developed after viewing results, use a new held-out test and label the previous run exploratory.

### Proposed success criteria (confirm after pilot, before final testing)

| Requirement | Provisional threshold |
|---|---|
| Mathematical gain | At least +5 percentage points over C0 on primary endpoint |
| Agent advantage | Positive advantage over both C1 and C4, with adjusted statistical evidence and effect sizes reported |
| Donor dependence | Meaningful advantage over C5; reversion reduces gain |
| Generalization | Improvement on an independent problem set or held-out templates |
| Capability retention | No more than 3 percentage-point loss on the preselected retention benchmark; report uncertainty |
| Structural constraint | Exactly unchanged parameter count, shapes and runtime graph; no gradient-based training |
| Device efficiency | Compare peak memory and latency under the same deployment format; target within 10% of base model, with uncertainty and device details |

A +5 point gain is a planning target, not a prediction. If only C0 is beaten, report improvement but do not claim agent superiority or genuine donor transfer. If performance rises mainly because looping decreases, report a behavior repair, not newly acquired mathematical reasoning. Negative or mixed outcomes remain publishable as an honest feasibility study if the evaluation is rigorous.

## 10. Remote donor weight access and storage plan

Kimi K3.1 Pro is an optional 2.8T-parameter mixture-of-experts donor with MXFP4 weights [11]. At two bytes per parameter, unquantized BF16 weights alone would occupy approximately 5.6 TB (decimal). Packed four-bit values have a theoretical 1.4 TB payload before scales, metadata and other tensors. Neither figure is a measured checkpoint size. Record actual shard sizes from the pinned official repository.

Hugging Face documents reading Safetensors metadata through small HTTP Range requests [12]. This makes targeted retrieval plausible, but the actual repository behavior, packed weight format and data usage must be tested. Remote retrieval still transfers requested bytes and does not provide access to full donor activations or expert routing during inference.

**Remote-access engineering acceptance checklist:**

- [ ] Pin checkpoint revision and retrieve configuration, license, tensor index and Safetensors metadata before any weight download.
- [ ] Test byte-range support on a tiny public Safetensors file, then a small shard of the actual donor; validate HTTP 206 and Content-Range. Reject unexpected full-body responses before downloading them.
- [ ] Account for the Safetensors header offset, tensor-relative data offsets, dtype, shape and contiguous layout. Test partial reads against a locally loaded small file.
- [ ] Verify expected content length, source revision and checksums where available. For partial data, log exact offsets and local hashes; do not claim full-file verification.
- [ ] Handle packed quantization and associated scales correctly; retrieve necessary companion tensors. Verify decoding with a small fixture before interpreting values.
- [ ] Provide bounded on-demand data access, a size-limited local cache, explicit byte budgets, retry limits and network usage logs. Do not cache credentials or expose tokens to the agent.
- [ ] Do not invoke full-model download/loading for the trillion-parameter donor. Disable automatic fallback to full downloads.
- [ ] Treat remote code as untrusted; prefer standard loaders and review any necessary custom code before execution.
- [ ] Compare remote access and local reference tensors on the small donor. Record bytes read, latency and cache hit rate.
- [ ] Keep weights-only results separate from any future activation-assisted runs. If weights alone cannot identify a useful component, record the limitation rather than silently switching methods.

**Initial planning limits:** 30 GB maximum local model/cache allocation for the first small-model stage (adjust after checking actual free space); 1 GB cumulative remote Kimi data for an optional metadata/access pilot, with no automatic escalation. Request researcher approval before exceeding limits. Kimi is a stretch goal, not an initial download requirement. Its 1 GB remote-data pilot limit is a proposed cap only, not permission to download. Use only pinned metadata and bounded tensor requests after separate approval. A normal inference API returns outputs, not arbitrary weight slices or internal activations.

## 11. Resource and timeline plan

| Period | Milestone | Exit gate |
|---|---|---|
| Week 1 | Literature review, README/repository setup, permissions, inspect hardware, freeze pilot plan | Written design and approved access |
| Week 2 | Evaluation harness and original WSP replication; validate remote reads on small files | Reproducible unmodified baseline and known-edit check |
| Week 3 | Matched agent competency pilot, if permitted | Agent selection and measured per-run cost |
| Weeks 4–5 | Qwen3.5-2B to Qwen3.5-0.8B feasibility experiment | Evidence of donor-dependent effect or documented failure |
| Weeks 6–7 | Qwen3.5-9B to Qwen3.5-0.8B if the first stage justifies it | Matched controls and independent runs within budget |
| Week 8 | Frozen evaluation, results and documentation | Feasibility report, with any unfinished confirmatory work identified |
| Months 3–5 (optional) | Expanded confirmation, cross-family experiment, optional Kimi remote pilot and paper | Separate approval and evidence-based progression |

These are planning estimates assuming roughly 10–15 research hours per week (80–120 hours over eight weeks). A negative result may end the study earlier; sufficient independent confirmatory runs may take longer. If the budget or time cap prevents confirmation, label results exploratory rather than reducing the standard of evidence.

### Initial study budget (USD)

| Expense | Proposed allocation |
|---|---:|
| Gemini API experiments | $30 |
| Astra API experiments | $60 |
| Cloud GPU computing | $40 |
| Storage and miscellaneous | $10 |
| Unallocated reserve | $60 |
| **Hard ceiling** | **$200** |

Expected actual spending: approximately $60–140 for the initial study, assuming most small-model inference runs locally. This is a planning range, not a quote or a measured workload. It excludes full Kimi inference or downloads and does not include the value of researcher time or existing hardware. The allocations are limits for planning, not permission to spend. A proposed $20 pilot is part of, not additional to, the $200 ceiling. Get explicit approval before starting paid work, then again before releasing further funds or reallocating the reserve.

**Budget controls:** verify live API and GPU rates before starting; calculate API cost as metered input/output tokens (including billable reasoning) times the applicable rates, plus any tool/cache charges. Estimate GPU charges using measured throughput and billed runtime. Record actual spend per run and cumulatively. Include taxes and storage/egress charges where applicable. Stop new paid jobs before projected total exceeds the authorized limit; use provider-side limits where available, and account for in-flight charges. Do not assume subscription access or campus computing is free, available, or permitted for this purpose. If the $20 pilot cannot establish a reliable baseline, pause and report costs rather than automatically scaling.

**Computing checklist:** record actual M4 Max unified memory, disk free space, accelerator support, model load footprint, inference speed, school cluster permissions/quotas and available GPU hours. Prefer local hardware and use school resources only with permission. Benchmark a small batch before committing to hundreds of long reasoning generations. Set per-run time, token, forward-pass, edit and trial limits; count actual examples and generated tokens. No paid resources or large downloads without approval.

### Kimi escalation gate (all required before a separate large-donor pilot)

- [ ] Reproduce an existing WSP result or establish an independently verified comparable baseline; document any replication failure before deciding whether to proceed.
- [ ] Demonstrate same-family AWST with at least a 5 percentage-point held-out gain over C0 and evidence of donor dependence and advantage over key controls.
- [ ] Replicate across independent agent runs and report uncertainty and regressions.
- [ ] Preserve parameter count, shapes and runtime graph with no gradient-based training.
- [ ] Complete a small cross-family feasibility experiment (Qwen to Gemma) before relying on Kimi-to-Gemma compatibility.
- [ ] Validate bounded remote tensor retrieval, quantization handling and provenance on small checkpoints.
- [ ] Obtain applicable agent/provider and model permissions, approve a separate Kimi budget and define a strict remote byte limit. No full checkpoint download or donor inference without separate explicit approval.

## 12. Repository template and research deliverables

Suggested layout (design, not an existing repository):

```text
awst/
  README.md
  AGENTS.md
  GEMINI.md
  .gitignore
  docs/
    progress.md
    handoffs/
    literature/
      bibliography.bib
      README.md
    research_protocol.md
    literature_review.md
    permissions.md
    decisions.md
    preregistration.md
  configs/
    models/
    experiments/
  src/
    data/
    evaluation/
    weight_io/
    interventions/
    baselines/
    agent_interface/
  tests/
  data/
    development/
    validation/
  runs/
    manifests/
    logs/
  results/
  paper/
```

Keep hidden tests and labels outside the agent-readable workspace, not merely in a folder named test. Exclude checkpoint binaries, secrets, private data, large caches and unnecessary logs from version control. The evaluator should expose development results only; final evaluation must be isolated. Record model IDs, revisions, seed, backend, prompts, patch provenance, changed parameter count, relative norm, resource budget, code commit, candidate selection and all results for each run. Store results in a machine-readable format and preserve run manifests and patch logs. Release code or checkpoints only when allowed by the applicable licenses and permissions.

**Expected outputs:** a literature review; an evaluation harness and verified remote-reader feasibility report; a baseline and control comparison; an empirical account of whether donor-specific transfer worked; reproducible experiment artifacts; and a research paper. A more capable mobile model is a possible outcome, not a promised deliverable. If a modified model succeeds, test deployment separately after quantization and on actual target hardware; do not infer iPhone compatibility from parameter count alone.

## 13. Paper outline (WIP)

1. Introduction: motivation, research questions, contributions and limitations.
2. Related work: WSP, weight merging, alignment, transfer and distillation.
3. Method: operational definitions, permitted agent role, resource budgets, controls and remote-access tracks.
4. Experimental setup: models, data, isolation, metrics, statistical analysis and reproducibility.
5. Results: complete tables, learning-free controls, donor dependency, negative results and costs. [Pending]
6. Analysis: causal evidence, alternative explanations, generalization and capability regressions. [Pending]
7. Limitations, safety, licensing and future work.
8. Conclusion: answer each research question based on evidence. [Pending]
9. References and appendix: prompts, configuration files, run manifests and detailed analysis.

## 14. Agent handoff / launch instructions

This is a research protocol, not authorization to run all proposed experiments. Treat decisions and thresholds as provisional until the researcher approves them.

### 14.1 Harness roles and manual handoff

- Antigravity is the primary implementation and research harness. Gemini 4 Neon or Carbon coordinates work, maintains the repository, performs permitted research and executes approved local tasks. Record the exact model and harness version in experiment records.
- Codex with GPT-6.2 Astra is a secondary specialist and, if authorized, a separate experimental weight programmer. Antigravity must not assume access to Codex or attempt to automate the user's account, credentials, quota or UI. The researcher manually submits each handoff prompt and returns the response or patch.
- For each handoff, create `docs/handoffs/HNNN_task.md` with the precise question, purpose, allowed operations, relevant file paths and commit SHA, model/checkpoint revisions, available tools, budget, constraints, acceptance tests, and expected output. Include the minimum necessary context and avoid hidden test data, credentials, or unnecessary raw weights. Specify whether the request is review-only or authorized implementation. Ask the researcher to send the prompt to GPT-6.2 Astra in Codex, then pause until they return the result.
- After receiving the response, log the actual model, prompt, result, relevant tool use, costs and attribution; review and test any proposed patch before integrating. Never assume an external recommendation is correct or that Codex has the same environment. Use a clean, explicitly identified commit or patch for cross-harness changes; reconcile concurrent changes rather than overwriting them.
- For a matched agent comparison, give each programmer equivalent starting information, tools and resource budgets in separate branches or workspaces. Do not let Gemini's analysis or Astra's edits contaminate the other agent's independent trial. Human-relayed prompts must follow a frozen template for comparisons. Distinguish collaborative engineering from experimental agent runs. No provider-hosted weight-improvement work until permission is resolved for the specific product and intended use.

### 14.2 Ordered project checklist and milestone exit gates

Proceed in this order. Update the checklist as evidence becomes available, and do not mark a task complete without recording its output and checks. A failed or blocked stage is an acceptable recorded outcome; do not silently skip it.

**M0: Literature and research setup**

- [ ] Read this protocol completely and inspect existing repository and hardware without destructive changes.
- [ ] Gather relevant primary research papers (start with section 15); search for newer work and competing methods. Save permitted papers or links in `docs/literature/`, with bibliographic metadata and exact versions. Do not redistribute copyrighted PDFs without permission.
- [ ] Read the methods, results and limitations; record each paper's relevance, evidence, gaps, and citation in `docs/literature_review.md` and `bibliography.bib`. Verify references rather than trusting this protocol's initial list. Reassess novelty and document any overlapping results.
- [ ] Set up basic README, repository structure, `.gitignore`, `AGENTS.md`, `GEMINI.md`, progress log, decision log, permissions log and task checklist. Verify approved remote, branch and write access; do not create a public repository or change visibility without approval.
- [ ] Verify exact checkpoint candidates, licenses, agent/provider terms, available compute, pricing, and storage. Prepare approval requests for unresolved items.
- [ ] Define narrow arithmetic diagnostic, controls, evaluation plan, feasibility criteria and proposed $20 pilot. Present for approval before paid work, checkpoint downloads or weight modifications.

**M1: Baseline and infrastructure**

- [ ] Set up pinned environment, reproducible configs, weight reader, evaluation harness and tests.
- [ ] Isolate final held-out data from both harnesses; verify scoring, tokenization and baseline reproducibility. Freeze the pilot protocol and record revisions.
- [ ] Validate selective tensor retrieval on small files; measure download bytes, storage and compute. Replicate a permitted WSP experiment or document failure.

**M2: Agent pilot and main experiments**

- [ ] Run approved small competency comparisons and measure actual cost; select programmer(s) using pilot evidence.
- [ ] Implement and test conventional transfer and matched controls; establish reproducible patch and provenance logs.
- [ ] Run the approved Qwen 2B to 0.8B AWST trials; log all outcomes, including failures.
- [ ] Review donor-dependence, preliminary results, budget and regressions before approving the Qwen 9B stage. Do not advance just because the checklist contains it.
- [ ] Run the approved second donor experiment if justified. Cross-family and Kimi work require the separate gates in section 11.

**M3: Verification and analysis**

- [ ] Freeze selected methods and checkpoints before final evaluation. Use the isolated evaluator with researcher involvement where needed.
- [ ] Verify reproducibility from recorded commits and manifests; compare matched controls, confidence intervals, independent runs, contamination risks and alternative explanations.
- [ ] Verify unchanged architecture and parameter count, absence of gradient training, provenance, retention, speed, memory and costs. Report failed runs and limitations.
- [ ] Request an independent review (Codex if permitted or another approved reviewer) without revealing hidden data; record and resolve findings. If evidence is insufficient, label the study exploratory or negative.

**M4: Writing and release preparation**

- [ ] Write the paper only after verified results, using section 13 and updating the WIP abstract to match evidence. Maintain citations throughout the project.
- [ ] Verify all figures, tables, numerical claims and references against stored outputs; document reproducibility instructions and limitations in README.
- [ ] Prepare final artifact manifest, expense summary and remaining work. Obtain human approval before publication, checkpoint distribution or public release.

### 14.3 Persistent context and recovery

Use `AGENTS.md` (plural, the conventional repository instruction filename) for stable operating rules shared by both harnesses. Use `GEMINI.md` as a short Antigravity/Gemini-specific entry point referencing `AGENTS.md` and the current protocol; verify actual instruction discovery in the installed harness and explicitly read these files at session start regardless. Do not rely on automatic loading. Avoid duplicating conflicting rules.

Keep the frequently updated state in `docs/progress.md`, not in an ever-growing instructions file. After every meaningful work unit, and before a handoff, compaction if detectable, session end, or approval pause, record:

- Date, milestone, completed task IDs, concise work summary and changed paths.
- Current branch, relevant commit SHA, test commands/results, run IDs and pointers to evidence.
- Decisions and rationale (link `docs/decisions.md`), current budget/actual spend and permissions status.
- Failed attempts, blockers, unanswered questions and exact next action; explicitly identify pending human approvals or handoffs.

Keep detailed experimental records in run manifests and logs; keep progress summaries concise. At session start or after compaction, read `AGENTS.md`, `GEMINI.md`, this protocol, `docs/progress.md`, pending handoffs and relevant decision/run records. Inspect `git status` and recent history, reconcile discrepancies, and resume from the first unfinished approved task. Do not repeat completed experiments or assume missing approvals. Record new commit SHAs after each commit, without creating an endless cycle of commits solely to record their own SHA.

### 14.4 Git workflow and human approval

After every completed major step or coherent milestone (M0–M4 may each contain several commits):

1. Update the progress log, checklist, decisions and any relevant README instructions. Run appropriate tests and inspect the diff and `git status`.
2. Stage only relevant reviewed files. Check for secrets, model binaries, hidden evaluation data, copyrighted papers, large logs and unrelated user changes. Keep a `.gitignore` and do not bypass safety checks.
3. Invoke `/git-commit` if this command exists in the active harness. Verify its behavior; if unavailable, use ordinary Git commands to perform the same reviewed commit and report the fallback. Use a short descriptive title (e.g., `Add baseline evaluator`). Do not double-commit if the command already committed.
4. Push to the approved remote and branch; verify success. Do not force-push, rewrite history, change remotes, publish private material or overwrite unrelated work without explicit approval. If authentication, conflicts, branch protection or network access blocks the push, record the local state and ask the researcher; do not claim success.
5. Continue to the next approved task once the commit and push succeed. No need to request approval for every routine commit/push after the researcher approves the remote and branch, but respect harness-required confirmations.

Pause and ask the researcher whenever human intervention is necessary: unresolved research choices, ambiguous instructions that materially affect results, model/provider permissions, paid work or budget changes, checkpoint downloads, weight modifications, external Codex handoffs, credentials/authentication, destructive operations, unexpected results requiring a protocol change, or any other approval gate. Present the specific decision, relevant evidence, recommendation and alternatives. Persist state before pausing; do not bypass approvals or continue the blocked action. Resume only after receiving the answer. Routine approved work may continue without unnecessary questions.

**Standing research rules:** Work incrementally, keep a reproducible decision log, isolate final tests, preserve original weights, and never hide failures. Do not invent evidence, claim benchmark gains before measurement, or change the protocol after seeing held-out results without documenting it. Stop at agreed budget or permission gates. Favor small, falsifiable experiments over scaling prematurely. The initial deliverable is a feasibility study design and a reproducible baseline, not a modified Kimi/Gemma checkpoint.

## 15. Initial references and technical resources

Primary and related research (verify bibliographic details and newer revisions before submission):

1. Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026, Nov. 2). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* https://rikiyat.github.io/blog/weight-space-programming/
2. Fan, J., Chen, S., Hou, Y., Zhang, A., & Xie, H. (2026). *Training-Free Knowledge Transfer Across Model Scales through Activation-Guided Pruning.* https://arxiv.org/abs/2608.13596
3. Cui, C., et al. (2026). *Transport and Merge: Cross-Architecture Merging for Large Language Models.* https://arxiv.org/abs/2602.05495
4. Li, B., et al. (2026). *WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming.* https://arxiv.org/abs/2611.25476
5. Ainsworth, S. K., Hayase, J., & Srinivasa, S. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* https://arxiv.org/abs/2209.04836
6. *Replicating and Extending Weight-Space Programming: Frontier LLMs Compile Programs into Weights and Read Input Layers* (2026). https://arxiv.org/abs/2611.21449
7. *ZipIt! Merging Models from Different Tasks without Training* (2024). https://proceedings.iclr.cc/paper_files/paper/2024/hash/7dbb5bfab324e3b86af9bd0df15498dd-Abstract-Conference.html

Model cards, engineering and terms:

8. Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
9. Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B
10. Gemma 4 E2B: https://huggingface.co/google/gemma-4-E2B-it
11. Kimi K3.1 Pro: https://huggingface.co/moonshotai/Kimi-K3.1-Pro
12. Hugging Face Safetensors metadata parsing: https://huggingface.co/docs/safetensors/en/metadata_parsing
13. OpenAI Terms of Use and Usage Policies: https://openai.com/policies/terms-of-use/ and https://openai.com/policies/usage-policies/
14. OpenAI Services Agreement: https://openai.com/policies/services-agreement/
15. Google Gemini API additional terms and Cloud service terms: https://ai.google.dev/gemini-api/terms and https://cloud.google.com/terms/service-terms
16. Candidate Qwen3.5-2B checkpoint (verify official model ID and revision before use): https://huggingface.co/Qwen/Qwen3.5-2B

---

**Researcher decisions still needed:** Final permitted agent and provider; exact arithmetic subskill; verified checkpoint revisions and compatibility; actual hardware/storage limits; pilot sample size; final success thresholds and statistical plan; explicit permission for the $20 pilot and any further paid or large-scale execution. The protocol proposes defaults where these are unknown. No model weights were downloaded, edited or evaluated in preparing this document.

**Revision log:** v0.2 narrows the scope to grade-school math; sets GSM8K as the primary confirmatory benchmark and synthetic arithmetic as a pilot/secondary diagnostic; specifies two staged Qwen donor candidates; narrows agent candidates; adds an eight-week plan, $200 ceiling and $20 pilot approval gate; and adds explicit Kimi escalation requirements. Existing literature references are retained pending verification during Phase 0.


**Revision log (v0.3):** Adopts Antigravity as primary harness with manual GPT-6.2 Astra/Codex handoffs; adds ordered milestone checklist, persistent progress and context recovery, concise commit/push workflow, and explicit pause/approval rules. Retains scientific design, spending gates and provisional results. Harness-specific command availability is not verified and must be checked locally.

# Permissions, Checkpoints, Pricing, and Compute Audit

**Last audited:** 2026-11-25  
**Spend to date:** \$0.00 / \$200.00 hard ceiling (proposed \$20.00 pilot gate **PENDING APPROVAL**)

---

## 1. Candidate Model Checkpoints and Licenses

> [!NOTE]
> All architecture fields below were verified from model-card/Hub metadata on 2026-11-25. Exact Hugging Face commit SHAs (`revision`) and raw `config.json` / `model.safetensors.index.json` files must be pinned and verified locally after M1 download approval.

| Model ID | Stage | License | Architecture & Key Verified Specs | Approx. Footprint (BF16 / Native) | Status |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | Stage 0 & 1 Recipient | **Apache 2.0** | `Qwen3_5ForConditionalGeneration`; multimodal vision-language; `24` text layers (`3:1` Gated DeltaNet + Full Attention, `full_attention_interval: 4`); `hidden_size: 1024`; `intermediate_size: 3584`; `num_attention_heads: 8`; ~0.87B params | ~1.5–1.75 GB | License clear; **download & SHA pinning pending M0 approval** |
| `Qwen/Qwen3.5-2B` | Stage 1 Donor | **Apache 2.0** | `Qwen3_5ForConditionalGeneration`; multimodal; `24` text layers (`full_attention_interval: 4`, same depth as 0.8B); `hidden_size: 2048`; `intermediate_size: 6144`; `vocab_size: 248320`; ~2B params | ~4.25 GB | License clear; **download & SHA pinning pending M0 approval** |
| `Qwen/Qwen3.5-9B` | Stage 2 Donor | **Apache 2.0** | `Qwen3_5ForConditionalGeneration`; multimodal; `32` text layers (`24` Gated DeltaNet + `8` Full Attention); `hidden_size: 4096`; `intermediate_size: 12288` | ~18–20 GB (verify exact shards) | License clear; **gated behind Stage 1 justification** |
| `google/gemma-4-E2B-it` | Stage 3 Recipient | **Apache 2.0** | `Gemma4ForConditionalGeneration`; multimodal any-to-any (text/image/audio); ~2B effective params; dense + Per-Layer Embeddings (PLE); 128K context | Verify exact shards before Stage 3 | License clear; **gated behind Stage 2** |
| `moonshotai/Kimi-K3.1-Pro` | Stage 4 Donor (Optional Stretch) | **Custom `kimi-k3.1-pro` License** | 2.8T total / 104B active MoE; 93 layers (`69` KDA + `24` Gated DSA); 896 routed + 2 shared experts (16 active); `MoonViT-V2` (401M); native MXFP4 (group 32, E8M0) for routed experts, BF16 elsewhere | ~1.56 TB across 96 shards | Research/modification permitted; commercial MaaS restrictions (>\$20M rev / attribution). **NO download; gated behind Kimi Escalation Gate** |

*Compatibility observation:* `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` share identical text layer count (`24`) and hybrid layout (`full_attention_interval: 4`), with `2B` having exact `2x` `hidden_size` (`2048` vs `1024`) and larger `intermediate_size` (`6144` vs `3584`), while both include multimodal vision encoders. Exact KV-head and Gated DeltaNet dimensions (`linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`) for `0.8B` vs `2B` must be verified directly from pinned `config.json` in M1.

---

## 2. Provider Terms of Service Gate (CRITICAL — UNRESOLVED)

Per `AWST_Research_Protocol_v0.3.md` Section 7, **no provider-hosted agent may be used for actual model-weight improvement until applicable terms are reviewed and resolved.**

| Provider / Product | Verified Restriction Clause | Affected AWST Work | Current Status |
|---|---|---|---|
| **OpenAI Terms of Use / Services Agreement** (`GPT-6.2 Astra` via Codex or API) | Prohibits using Output *"to develop models that compete with OpenAI"* (or artificial intelligence models competing with OpenAI's products/services). | Using Astra/Codex to generate or execute edits that improve `Qwen3.5` or `Gemma 4` weights (e.g., Stage 0 WSP replication, `C1`, `T1`, `T2`). | **BLOCKED for model-improvement runs** until researcher resolution. Review-only protocol/infra work and synthetic toy tests are distinct. |
| **Google Gemini API Additional Terms** (`Gemini 4 Neon` / `Carbon` via Antigravity or API) | Prohibits using Services *"to develop models that compete with the Services (e.g., Gemini API or Google AI Studio)"* or attempting to reverse engineer, extract, or replicate underlying models. | Using Gemini 4 Neon/Carbon directly as a weight-space programmer (`C1`/`T1`/`T2`) to improve open-weight LLMs. | **BLOCKED for model-improvement runs** until researcher resolution. Infrastructure coding, evaluation harness, documentation, and coordination are permitted. |

### Required Researcher Decision / Resolution Paths
1. **Institutional / Provider Clarification:** Obtain written clarification or confirm institutional agreement confirming that non-commercial, academic interpretability/feasibility research modifying small open-weight checkpoints (`0.8B`–`2B`) does not constitute developing a competing commercial model/service.
2. **Explicit Researcher Compliance Determination:** Researcher reviews applicable account terms (consumer/Pro vs API/Enterprise/academic) and records an explicit written determination in `docs/permissions.md` before authorizing provider-hosted weight edits.
3. **Open-Weight Fallback Programmer:** If provider ToS prohibits using Astra or Gemini to edit open-weight LLMs, use a suitably licensed local or hosted open-weight coding/reasoning agent for the weight-improvement arms (`C1`, `T1`, `T2`), while keeping Antigravity/Codex strictly to research infrastructure, evaluation, and analysis.

---

## 3. Verified Live API Pricing (Checked 2026-11-25)

| Model | Input (/ 1M tokens) | Cached Input (/ 1M) | Output (/ 1M tokens) | Billing Notes |
|---|---|---|---|---|
| **GPT-6.2 Astra** (API) | \$8.00 | \$0.40 | \$40.00 | Background Mode: -30% (2h) / -40% (8h); Fast ~2x, Ultrafast 3x–6x; regional +10%. Codex app/CLI subscription runs must also be shadow-costed against API rates and usage caps. |
| **GPT-6.2 Sol** (API) | \$2.00 | \$0.10 | \$10.00 | Lower-cost OpenAI tier. |
| **Gemini 4 Neon** (API) | \$3.00 | N/A (verify caching) | \$15.00 | Output rate includes thinking tokens; **no Free Tier**. |
| **Gemini 4 Carbon** (API) | \$2.00 (promo thru 2026-11-30) / \$4.00 (from 2026-12-01) | N/A | \$10.00 (promo) / \$20.00 (from 2026-12-01) | Output rate includes thinking tokens; **no Free Tier**. |

*Note on Codex `GPT-6.2 Astra` vs raw API:* OpenAI safety evaluations noted Codex App/CLI includes built-in system prompt mitigations for blocker reporting that raw API integrations do not apply automatically; record exact harness (`Codex` vs `Responses API`) and reasoning effort for every run.

---

## 4. Verified Local Hardware, Storage, and Software Environment

| Resource | Measured Value (2026-11-25) | Implication |
|---|---|---|
| **Host / OS** | Apple M4 Max, macOS `27.2.1` (`26B131`) | Local execution viable for `0.8B` and `2B` BF16 inference; `9B` (~18–20 GB BF16) is tight alongside OS overhead on 36 GiB unified memory. |
| **Memory & CPU** | `38,654,705,664` bytes (**36 GiB** unified memory), **14 CPU cores** | Benchmark `0.8B` & `2B` peak memory and tok/s on a 10-problem micro-batch in M1 before full runs. |
| **Disk Space** | `926 GiB` total, **580 GiB available** on `/` | 30 GB local cache cap for Stages 0–2 uses ~5.2% of free disk space. |
| **Python Binaries** | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` (`3.14.5`), `/opt/homebrew/bin/python3.11` | `huggingface_hub 1.29.0`, `numpy 2.5.1`, `matplotlib 3.11.1`, `httpx 0.28.1` exist in Python 3.14, but **NO ML stack (`torch`, `mlx`, `transformers`, `safetensors`, `datasets`, `scipy`, `pytest`) is installed**. |
| **Environment Action (M1)** | Pending M0 approval | Create project-local `.venv` (gitignored), checking wheel support for PyTorch MPS + `transformers>=5.2` (required for `qwen3_5`) across Python 3.11 vs 3.14. |
| **School Cluster / Cloud GPU** | Not yet audited or authorized | Prefer local M4 Max first; \$40 cloud GPU budget remains locked unless local throughput blocks progress and researcher approves. |

---

## 5. Git Remote and Repository Access

- **Remote:** `origin` = `https://github.com/ItsKaiwenDu/awst.git`
- **Branch:** `main` (tracking `origin/main`)
- **Visibility:** Not queried via GitHub API (`gh` CLI not installed); per protocol, do **not** create a public repository or modify repository visibility without researcher approval.

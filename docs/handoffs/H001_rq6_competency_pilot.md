# H001: RQ6 Matched Weight-Space Competency Pilot (Synthetic Toy Networks)

- **Date:** 2026-11-26
- **Target Agent / Harness:** GPT-6.2 Astra via Codex
- **Authorization Mode:** `AUTHORIZED IMPLEMENTATION (Diagnostic Toy Networks Only — $0.00 LLM Weight Modification)`
- **Repository Commit SHA:** `380b7b9` (update to latest M2a-2 SHA before submission if committed after `run_m2_baselines`)
- **Model / Checkpoint Revisions:** N/A (synthetic `.safetensors` toy networks in `data/competency_pilot/`)
- **Budget / Resource Limits:** `<= $4.00` shadow API cost; max `1` Python analysis/patching script; no gradient descent/training; no external network calls.
- **Relevant Files (Read-Only):**
  - `data/competency_pilot/suite_manifest.json`
  - `data/competency_pilot/task_a_read_mlp/model.safetensors`
  - `data/competency_pilot/task_b_transplant_rotated/donor.safetensors`
  - `data/competency_pilot/task_b_transplant_rotated/recipient.safetensors`
  - `data/competency_pilot/task_c_localize_corruption/corrupted.safetensors`
- **Strictly Forbidden Files (Isolation Rule for RQ6):**
  - Do **NOT** read `src/agent_interface/competency_suite.py` or `tests/test_agent_interface.py` (they contain the generator code and verifier).
  - Do **NOT** access `/Users/kaiwendu/.awst_isolated_eval/`.

> **Setup Note (Run First if `.safetensors` Files Are Not Yet Present on Disk):**  
> Because `*.safetensors` are gitignored, ensure the toy files exist by running once before starting the agent session:  
> `.venv/bin/python -c "from src.agent_interface.competency_suite import generate_competency_suite; generate_competency_suite()"`

---

## 1. Purpose and Objective

Run a controlled, matched-prompt diagnostic (`RQ6` / Protocol Section 7) testing whether GPT-6.2 Astra can:
1. **Task A (`task_a_read_mlp`):** Interpret raw neural network weights without input/output labels.
2. **Task B (`task_b_transplant_rotated`):** Perform basis-aligned, training-free donor-to-recipient weight transplantation across mismatched hidden widths (`32 -> 16`) and orthogonal basis rotations.
3. **Task C (`task_c_localize_corruption`):** Localize and repair a planted sparse weight corruption in a 3-layer ReLU MLP.

Your submission will be scored automatically by `verify_competency_submission(...)` and compared against an isolated Gemini 4 Neon trial run under the exact same prompt and file restrictions.

---

## 2. Tasks (Complete All Three)

### Task A — Weight Inspection (`data/competency_pilot/task_a_read_mlp/model.safetensors`)
- **Architecture:** `y = fc2(relu(fc1(x)))` (`input_dim=6, hidden_dim=16, output_dim=2`).
- **Objective:** Inspect only the raw `.safetensors` weights (`fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`).
  1. State the exact mathematical formulas computed by `output[0]` and `output[1]` for non-negative inputs `x[0], x[1]` (with `x[2:6] = 0`).
  2. List the exact integer indices of the active hidden neurons in `fc1`.

### Task B — Basis-Rotated Weight Transplantation (`data/competency_pilot/task_b_transplant_rotated/`)
- **Files:** `donor.safetensors` (`8 -> 32 -> 4`) and `recipient.safetensors` (`8 -> 16 -> 4`).
- **Architecture:** 2-layer linear network `y = fc2(fc1(x) + b1) + b2`.
- **Objective:** Donor computes a 4-class linear map embedded in a basis-rotated 32-dimensional hidden space. Write a **training-free** closed-form Python patch (no gradient descent, no data fitting, no forward-pass optimization) that writes adapted donor weights into `recipient`'s existing `(16, 8)` and `(4, 16)` parameter slots and saves the result to:
  `runs/competency_astra/task_b_patched.safetensors` (preserving all keys, shapes, and `float32` dtype).

### Task C — Corruption Localization & Repair (`data/competency_pilot/task_c_localize_corruption/corrupted.safetensors`)
- **Architecture:** `y = fc3(relu(fc2(relu(fc1(x)))))` (`16 -> 32 -> 32 -> 4`).
- **Objective:** Exactly 4 columns of one weight matrix contain a planted outlier scale + sign corruption (`-25x`). Without training or external data:
  1. Identify the exact tensor name (e.g., `fc2.weight`) and the exact 4 integer column indices.
  2. Write a sparse closed-form repair patch restoring those 4 columns and save the repaired checkpoint to:
     `runs/competency_astra/task_c_repaired.safetensors`.

---

## 3. Expected Deliverable Format

Return **one** concise markdown report containing:
1. **Task A Answer:** Exact formulas for `output[0]`, `output[1]`, and JSON list `task_a_active_neurons: [...]`.
2. **Task B Method & Artifact:** Explanation of your algebraic transplantation formula and confirmation that `runs/competency_astra/task_b_patched.safetensors` was written (or exact Python script to produce it).
3. **Task C Answer & Artifact:** Exact `task_c_tensor_name`, JSON list `task_c_cols: [...]`, repair rule, and confirmation that `runs/competency_astra/task_c_repaired.safetensors` was written (or exact Python script to produce it).
4. **Resource Accounting:** Exact token count / Codex turn count / shadow USD cost if visible.

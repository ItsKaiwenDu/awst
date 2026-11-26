"""RQ6 synthetic toy-network competency suite and deterministic verifier (non-LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def _relu_mlp_forward(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    h = F.relu(x @ weights["fc1.weight"].T + weights["fc1.bias"])
    if "fc2.weight" in weights and "fc3.weight" in weights:
        h = F.relu(h @ weights["fc2.weight"].T + weights["fc2.bias"])
        return h @ weights["fc3.weight"].T + weights["fc3.bias"]
    return h @ weights["fc2.weight"].T + weights["fc2.bias"]


def generate_competency_suite(
    out_dir: str | Path = "data/competency_pilot",
    seed: int = 2026,
) -> dict[str, Any]:
    """Create 3 deterministic toy-network WSP diagnostic tasks in Safetensors format."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------
    # Task A: Inspect weights of a 2-layer ReLU MLP (6 -> 16 -> 2)
    # Encodes exact rule: given inputs [a, b, 0, 0, 0, 0] with a,b in [0..9],
    # output[0] = a + b, output[1] = relu(a + b - 9) = max(0, a + b - 9) (>0 iff carry!)
    # ------------------------------------------------------------------
    w1_a = torch.zeros((16, 6), dtype=torch.float32)
    b1_a = torch.zeros((16,), dtype=torch.float32)
    w2_a = torch.zeros((2, 16), dtype=torch.float32)
    b2_a = torch.zeros((2,), dtype=torch.float32)
    # Neuron 0 computes relu(x0 + x1) -> sum
    w1_a[0, 0] = 1.0
    w1_a[0, 1] = 1.0
    # Neuron 3 computes relu(x0 + x1 - 9) -> carry surplus
    w1_a[3, 0] = 1.0
    w1_a[3, 1] = 1.0
    b1_a[3] = -9.0
    # Readouts
    w2_a[0, 0] = 1.0
    w2_a[1, 3] = 1.0
    task_a_dir = root / "task_a_read_mlp"
    task_a_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"fc1.weight": w1_a, "fc1.bias": b1_a, "fc2.weight": w2_a, "fc2.bias": b2_a},
        str(task_a_dir / "model.safetensors"),
    )

    # ------------------------------------------------------------------
    # Task B: Basis-rotated donor-to-recipient transplantation (8 -> 32 -> 4 into 8 -> 16 -> 4)
    # Donor computes y = W_true @ x using an exact rank-4 subspace embedded in 32 hidden units
    # rotated by an orthogonal matrix Q_32 and input-permuted vs recipient!
    # Unaligned slice copying fails (~0.25 acc / high MSE).
    # ------------------------------------------------------------------
    q_mat, _ = torch.linalg.qr(torch.randn((32, 32), generator=g, dtype=torch.float32))
    # True target: 4-class linear classifier embedded in first 4 ReLU pairs (8 dims) of donor
    w_base = torch.randn((4, 8), generator=g, dtype=torch.float32)
    w1_canon = torch.zeros((32, 8), dtype=torch.float32)
    w1_canon[:4, :] = w_base
    w1_canon[4:8, :] = -w_base
    w2_canon = torch.zeros((4, 32), dtype=torch.float32)
    w2_canon[:, :4] = torch.eye(4)
    w2_canon[:, 4:8] = -torch.eye(4)

    # Rotate hidden basis by Q and permute/scale donor hidden space:
    # W1_donor = Q @ W1_canon; W2_donor = W2_canon @ Q^T
    # Notice: for linear-equivalent active ReLU pairs or exact orthogonal transform?
    # Note: ReLU does NOT commute with dense orthogonal Q! Instead use a clean invertible
    # permutation + positive diagonal scaling on active units + dense rotation on a linear readout
    # OR make the task an exact linear residual block / linear-gated circuit where Q DOES commute!
    # Better: make donor a 2-layer LINEAR residual block (no ReLU between fc1/fc2): y = W2 @ (W1 @ x + b1) + b2
    # where hidden_donor=32 is rotated by Q_32 and hidden_rec=16 is initialized randomly!
    # Wait: that tests true basis-invariant SVD / rank-8 Procrustes transplantation!
    w1_don = q_mat @ w1_canon
    b1_don = torch.zeros((32,), dtype=torch.float32)
    w2_don = w2_canon @ q_mat.T
    b2_don = torch.zeros((4,), dtype=torch.float32)

    w1_rec = 0.05 * torch.randn((16, 8), generator=g, dtype=torch.float32)
    b1_rec = torch.zeros((16,), dtype=torch.float32)
    w2_rec = 0.05 * torch.randn((4, 16), generator=g, dtype=torch.float32)
    b2_rec = torch.zeros((4,), dtype=torch.float32)

    task_b_dir = root / "task_b_transplant_rotated"
    task_b_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"fc1.weight": w1_don, "fc1.bias": b1_don, "fc2.weight": w2_don, "fc2.bias": b2_don},
        str(task_b_dir / "donor.safetensors"),
    )
    save_file(
        {"fc1.weight": w1_rec, "fc1.bias": b1_rec, "fc2.weight": w2_rec, "fc2.bias": b2_rec},
        str(task_b_dir / "recipient.safetensors"),
    )

    # ------------------------------------------------------------------
    # Task C: Localize and repair a planted 4-column sign-flip + 10x corruption in 3-layer MLP
    # ------------------------------------------------------------------
    w1_c = torch.randn((32, 16), generator=g, dtype=torch.float32) * 0.4
    b1_c = torch.zeros((32,), dtype=torch.float32)
    w2_c = torch.randn((32, 32), generator=g, dtype=torch.float32) * 0.4
    b2_c = torch.zeros((32,), dtype=torch.float32)
    w3_c = torch.randn((4, 32), generator=g, dtype=torch.float32) * 0.4
    b3_c = torch.zeros((4,), dtype=torch.float32)

    corrupt_cols = [5, 11, 19, 27]
    w2_corrupted = w2_c.clone()
    # Plant strong outlier sign-flip + 25x norm spike on 4 columns of fc2.weight
    w2_corrupted[:, corrupt_cols] = -25.0 * w2_c[:, corrupt_cols]

    task_c_dir = root / "task_c_localize_corruption"
    task_c_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "fc1.weight": w1_c,
            "fc1.bias": b1_c,
            "fc2.weight": w2_corrupted,
            "fc2.bias": b2_c,
            "fc3.weight": w3_c,
            "fc3.bias": b3_c,
        },
        str(task_c_dir / "corrupted.safetensors"),
    )
    # Also save reference clean weights inside a private verification blob or regenerate from seed=2026
    spec = {
        "seed": seed,
        "tasks": {
            "task_a": {
                "file": str(task_a_dir / "model.safetensors"),
                "architecture": "2-layer ReLU MLP: y = fc2(relu(fc1(x)))",
                "prompt": "Inspect weights only. Describe the exact mathematical function computed by output[0] and output[1] on non-negative inputs x[0], x[1], and identify which hidden neuron indices are active.",
            },
            "task_b": {
                "donor_file": str(task_b_dir / "donor.safetensors"),
                "recipient_file": str(task_b_dir / "recipient.safetensors"),
                "architecture": "2-layer linear network: y = fc2(fc1(x) + b1) + b2 (donor hidden=32, recipient hidden=16)",
                "prompt": "Donor computes a 4-class linear map rotated into a 32-dim hidden space. Write a training-free weight patch copying/adapting the donor map into recipient's (16, 8) and (4, 16) slots so recipient matches donor outputs.",
            },
            "task_c": {
                "corrupted_file": str(task_c_dir / "corrupted.safetensors"),
                "architecture": "3-layer ReLU MLP: y = fc3(relu(fc2(relu(fc1(x)))))",
                "prompt": "4 columns in one weight matrix have a planted outlier scale+sign corruption (-25x). Identify the exact tensor name and 4 column indices, and write a sparse closed-form repair patch restoring the network.",
            },
        },
    }
    (root / "suite_manifest.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def verify_competency_submission(
    suite_dir: str | Path,
    task_a_active_neurons: list[int],
    task_b_patched_recipient_path: str | Path,
    task_c_tensor_name: str,
    task_c_cols: list[int],
    task_c_repaired_path: str | Path,
    seed: int = 2026,
) -> dict[str, Any]:
    """Deterministically score competency tasks A, B, and C without human subjectivity."""
    root = Path(suite_dir)
    g = torch.Generator().manual_seed(seed + 99)

    # Task A check: active neurons {0, 3} identified & forward test passes
    w_a = load_file(str(root / "task_a_read_mlp/model.safetensors"))
    x_a = torch.randint(0, 10, (50, 6), generator=g).to(torch.float32)
    y_a = _relu_mlp_forward(x_a, w_a)
    expected_sum = x_a[:, 0] + x_a[:, 1]
    expected_carry_surplus = torch.clamp(expected_sum - 9.0, min=0.0)
    assert torch.allclose(y_a[:, 0], expected_sum, atol=1e-4)
    assert torch.allclose(y_a[:, 1], expected_carry_surplus, atol=1e-4)
    task_a_ok = set(task_a_active_neurons) == {0, 3}

    # Task B check: patched recipient matches donor on 200 random Gaussian inputs
    w_b_don = load_file(str(root / "task_b_transplant_rotated/donor.safetensors"))
    w_b_pat = load_file(str(task_b_patched_recipient_path))
    assert list(w_b_pat["fc1.weight"].shape) == [16, 8]
    assert list(w_b_pat["fc2.weight"].shape) == [4, 16]
    x_b = torch.randn((200, 8), generator=g, dtype=torch.float32)
    y_don = (x_b @ w_b_don["fc1.weight"].T + w_b_don["fc1.bias"]) @ w_b_don["fc2.weight"].T + w_b_don["fc2.bias"]
    y_pat = (x_b @ w_b_pat["fc1.weight"].T + w_b_pat["fc1.bias"]) @ w_b_pat["fc2.weight"].T + w_b_pat["fc2.bias"]
    don_cls = torch.argmax(y_don, dim=-1)
    pat_cls = torch.argmax(y_pat, dim=-1)
    task_b_acc = float((don_cls == pat_cls).to(torch.float32).mean().item())
    task_b_rel_err = float(
        (torch.linalg.vector_norm(y_pat - y_don) / torch.linalg.vector_norm(y_don)).item()
    )

    # Task C check: regenerate clean reference from seed and verify localization + restoration
    g_ref = torch.Generator().manual_seed(seed)
    _ = torch.linalg.qr(torch.randn((32, 32), generator=g_ref, dtype=torch.float32))
    _ = torch.randn((4, 8), generator=g_ref, dtype=torch.float32)
    _ = 0.05 * torch.randn((16, 8), generator=g_ref, dtype=torch.float32)
    _ = 0.05 * torch.randn((4, 16), generator=g_ref, dtype=torch.float32)
    w1_c = torch.randn((32, 16), generator=g_ref, dtype=torch.float32) * 0.4
    w2_c = torch.randn((32, 32), generator=g_ref, dtype=torch.float32) * 0.4
    w3_c = torch.randn((4, 32), generator=g_ref, dtype=torch.float32) * 0.4
    clean_c = {
        "fc1.weight": w1_c,
        "fc1.bias": torch.zeros((32,), dtype=torch.float32),
        "fc2.weight": w2_c,
        "fc2.bias": torch.zeros((32,), dtype=torch.float32),
        "fc3.weight": w3_c,
        "fc3.bias": torch.zeros((4,), dtype=torch.float32),
    }
    w_c_rep = load_file(str(task_c_repaired_path))
    x_c = torch.randn((200, 16), generator=g, dtype=torch.float32)
    y_c_clean = _relu_mlp_forward(x_c, clean_c)
    y_c_rep = _relu_mlp_forward(x_c, w_c_rep)
    task_c_acc = float((torch.argmax(y_c_clean, dim=-1) == torch.argmax(y_c_rep, dim=-1)).to(torch.float32).mean().item())
    task_c_loc_ok = (task_c_tensor_name == "fc2.weight") and (set(task_c_cols) == {5, 11, 19, 27})

    return {
        "task_a_passed": task_a_ok,
        "task_b_agreement": round(task_b_acc, 4),
        "task_b_rel_l2_error": round(task_b_rel_err, 6),
        "task_b_passed": task_b_acc >= 0.95,
        "task_c_localized": task_c_loc_ok,
        "task_c_agreement": round(task_c_acc, 4),
        "task_c_passed": task_c_loc_ok and (task_c_acc >= 0.99),
        "overall_score": int(task_a_ok) + int(task_b_acc >= 0.95) + int(task_c_loc_ok and task_c_acc >= 0.99),
    }

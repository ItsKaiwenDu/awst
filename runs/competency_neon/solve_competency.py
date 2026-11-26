"""Training-free closed-form solver for RQ6 competency pilot (Tasks A, B, C)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def solve_task_a(model_path: Path) -> dict:
    tensors = load_file(str(model_path))
    w1 = tensors["fc1.weight"]
    b1 = tensors["fc1.bias"]
    w2 = tensors["fc2.weight"]
    b2 = tensors["fc2.bias"]

    # Active hidden neurons: non-zero incoming weights/bias AND non-zero outgoing weights
    incoming_active = (w1.abs().sum(dim=1) > 0) | (b1.abs() > 0)
    outgoing_active = w2.abs().sum(dim=0) > 0
    active_neurons = torch.where(incoming_active & outgoing_active)[0].tolist()

    # Verify numerically on non-negative grid for x[0], x[1] with x[2:6] = 0
    grid = torch.linspace(0.0, 20.0, 21)
    xs = []
    for x0 in grid:
        for x1 in grid:
            xs.append([x0.item(), x1.item(), 0.0, 0.0, 0.0, 0.0])
    x = torch.tensor(xs, dtype=torch.float32)
    h = torch.relu(x @ w1.T + b1)
    y = h @ w2.T + b2

    expected_y0 = x[:, 0] + x[:, 1]
    expected_y1 = torch.relu(x[:, 0] + x[:, 1] - 9.0)
    max_err_0 = (y[:, 0] - expected_y0).abs().max().item()
    max_err_1 = (y[:, 1] - expected_y1).abs().max().item()
    assert max_err_0 == 0.0 and max_err_1 == 0.0, f"Formula mismatch: {max_err_0}, {max_err_1}"

    return {
        "task_a_active_neurons": active_neurons,
        "output_0_formula": "output[0] = relu(x[0] + x[1]) = x[0] + x[1] (for x[0], x[1] >= 0)",
        "output_1_formula": "output[1] = relu(x[0] + x[1] - 9) = max(0, x[0] + x[1] - 9)",
        "verification_max_abs_err": [max_err_0, max_err_1],
    }


def solve_task_b(donor_path: Path, recipient_path: Path, out_path: Path) -> dict:
    donor = load_file(str(donor_path))
    recipient = load_file(str(recipient_path))

    w1_d = donor["fc1.weight"].to(torch.float64)  # (32, 8)
    b1_d = donor["fc1.bias"].to(torch.float64)    # (32,)
    w2_d = donor["fc2.weight"].to(torch.float64)  # (4, 32)
    b2_d = donor["fc2.bias"].to(torch.float64)    # (4,)

    # Effective affine map of the 2-layer linear network:
    # y = W2_d (W1_d x + b1_d) + b2_d = (W2_d W1_d) x + (W2_d b1_d + b2_d)
    m_eff = w2_d @ w1_d                  # (4, 8)
    b_eff = w2_d @ b1_d + b2_d           # (4,)

    # Closed-form balanced SVD factorization into recipient hidden dim (16 >= rank <= 4)
    u, s, vh = torch.linalg.svd(m_eff, full_matrices=False)  # u:(4,4), s:(4,), vh:(4,8)
    rank = s.shape[0]
    h_rec = recipient["fc1.weight"].shape[0]
    in_dim = recipient["fc1.weight"].shape[1]
    out_dim = recipient["fc2.weight"].shape[0]
    assert h_rec >= rank

    sqrt_s = torch.sqrt(s)
    w1_r = torch.zeros((h_rec, in_dim), dtype=torch.float64)
    w2_r = torch.zeros((out_dim, h_rec), dtype=torch.float64)
    w1_r[:rank, :] = sqrt_s[:, None] * vh
    w2_r[:, :rank] = u * sqrt_s[None, :]
    b1_r = torch.zeros_like(recipient["fc1.bias"], dtype=torch.float64)
    b2_r = b_eff

    patched = {
        "fc1.weight": w1_r.to(recipient["fc1.weight"].dtype).contiguous(),
        "fc1.bias": b1_r.to(recipient["fc1.bias"].dtype).contiguous(),
        "fc2.weight": w2_r.to(recipient["fc2.weight"].dtype).contiguous(),
        "fc2.bias": b2_r.to(recipient["fc2.bias"].dtype).contiguous(),
    }

    # Verify exact key, shape, and dtype preservation
    assert set(patched.keys()) == set(recipient.keys())
    for k in recipient:
        assert patched[k].shape == recipient[k].shape, f"Shape mismatch on {k}"
        assert patched[k].dtype == recipient[k].dtype == torch.float32, f"Dtype mismatch on {k}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(patched, str(out_path))

    # Reload from disk and verify functional equivalence against donor in float32
    reloaded = load_file(str(out_path))
    gen = torch.Generator().manual_seed(0)
    x_test = torch.randn((1024, in_dim), generator=gen, dtype=torch.float32)
    y_donor = (x_test @ donor["fc1.weight"].T + donor["fc1.bias"]) @ donor["fc2.weight"].T + donor["fc2.bias"]
    y_rec = (x_test @ reloaded["fc1.weight"].T + reloaded["fc1.bias"]) @ reloaded["fc2.weight"].T + reloaded["fc2.bias"]

    max_abs_err = (y_rec - y_donor).abs().max().item()
    rel_frob_err = ((reloaded["fc2.weight"] @ reloaded["fc1.weight"] - donor["fc2.weight"] @ donor["fc1.weight"]).norm()
                    / (donor["fc2.weight"] @ donor["fc1.weight"]).norm()).item()
    argmax_match = (y_rec.argmax(dim=-1) == y_donor.argmax(dim=-1)).float().mean().item()

    return {
        "saved_path": str(out_path),
        "donor_effective_singular_values": [round(v, 6) for v in s.tolist()],
        "max_output_abs_err": max_abs_err,
        "effective_matrix_rel_frob_err": rel_frob_err,
        "argmax_agreement_1024": argmax_match,
    }


def solve_task_c(corrupted_path: Path, out_path: Path) -> dict:
    tensors = load_file(str(corrupted_path))

    # Detect tensor and columns with extreme outlier column L2 norms (~25x median)
    outliers = []
    for name, t in tensors.items():
        if t.ndim != 2:
            continue
        col_norms = t.norm(dim=0)
        med = col_norms.median().item()
        ratios = col_norms / med
        bad = torch.where(ratios > 10.0)[0].tolist()
        if bad:
            outliers.append((name, bad, med, [round(col_norms[i].item(), 4) for i in bad], [round(ratios[i].item(), 2) for i in bad]))

    assert len(outliers) == 1, f"Expected 1 corrupted tensor, found: {outliers}"
    tensor_name, cols, med_norm, bad_norms, bad_ratios = outliers[0]
    cols = sorted(cols)
    assert len(cols) == 4, f"Expected 4 corrupted columns, found: {cols}"

    # Sparse closed-form repair: invert the planted -25x scale+sign corruption on those 4 columns
    repaired = {k: v.clone() for k, v in tensors.items()}
    w = repaired[tensor_name].to(torch.float64)
    w[:, cols] = w[:, cols] / -25.0
    repaired[tensor_name] = w.to(tensors[tensor_name].dtype).contiguous()

    # Verify exact key, shape, dtype preservation and sparsity of edit
    assert set(repaired.keys()) == set(tensors.keys())
    changed_params = 0
    for k in tensors:
        assert repaired[k].shape == tensors[k].shape
        assert repaired[k].dtype == tensors[k].dtype == torch.float32
        diff_mask = repaired[k] != tensors[k]
        if k != tensor_name:
            assert not diff_mask.any(), f"Unexpected change in {k}"
        else:
            changed_params = int(diff_mask.sum().item())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(repaired, str(out_path))

    reloaded = load_file(str(out_path))
    repaired_col_norms = reloaded[tensor_name].norm(dim=0)
    return {
        "task_c_tensor_name": tensor_name,
        "task_c_cols": cols,
        "saved_path": str(out_path),
        "pre_repair_median_col_norm": round(med_norm, 4),
        "pre_repair_corrupted_col_norms": bad_norms,
        "pre_repair_ratios_to_median": bad_ratios,
        "post_repair_col_norms_for_repaired_cols": [round(repaired_col_norms[i].item(), 4) for i in cols],
        "post_repair_all_col_norms_min_med_max": [
            round(repaired_col_norms.min().item(), 4),
            round(repaired_col_norms.median().item(), 4),
            round(repaired_col_norms.max().item(), 4),
        ],
        "changed_param_count": changed_params,
    }


def main() -> None:
    root = Path("/Users/kaiwendu/Documents/GitHub/awst")
    out_dir = root / "runs" / "competency_neon"
    out_dir.mkdir(parents=True, exist_ok=True)

    res_a = solve_task_a(root / "data/competency_pilot/task_a_read_mlp/model.safetensors")
    res_b = solve_task_b(
        root / "data/competency_pilot/task_b_transplant_rotated/donor.safetensors",
        root / "data/competency_pilot/task_b_transplant_rotated/recipient.safetensors",
        out_dir / "task_b_patched.safetensors",
    )
    res_c = solve_task_c(
        root / "data/competency_pilot/task_c_localize_corruption/corrupted.safetensors",
        out_dir / "task_c_repaired.safetensors",
    )

    summary = {"task_a": res_a, "task_b": res_b, "task_c": res_c}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

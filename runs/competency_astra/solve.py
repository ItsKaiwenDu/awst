"""Reproduce H001 from the four local fixtures. Run from a fixture workspace."""

import hashlib
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import safetensors
from safetensors import safe_open
from safetensors.numpy import save_file


DATA = Path("data/competency_pilot")
OUT = Path("runs/competency_astra")
W1, B1, W2, B2 = "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with safe_open(path, framework="np") as archive:
        return {key: archive.get_tensor(key) for key in archive.keys()}, archive.metadata()


def inspect_relu(weights):
    # A row can take a positive value on this orthant iff its bias or a
    # coefficient on one of the two unbounded inputs is strictly positive.
    directions = weights[W1][:, :2]
    active = np.flatnonzero((directions > 0).any(axis=1) | (weights[B1] > 0))
    return {
        "task_a_active_neurons": active.tolist(),
        "nonzero_fc1_rows": np.flatnonzero((weights[W1] != 0).any(axis=1)).tolist(),
        "fc2_bias": weights[B2].tolist(),
        "active_paths": [
            {
                "neuron": int(i),
                "x0_x1": directions[i].tolist(),
                "bias": float(weights[B1][i]),
                "to_outputs": weights[W2][:, i].tolist(),
            }
            for i in active if np.any(weights[W2][:, i])
        ],
    }


def affine(weights):
    left = weights[W2].astype(np.float64)
    return left @ weights[W1].astype(np.float64), left @ weights[B1].astype(np.float64) + weights[B2]


def transplant(donor, recipient):
    matrix, offset = affine(donor)
    width = matrix.shape[1]
    patch = dict(recipient)
    patch[W1] = np.zeros_like(recipient[W1])
    patch[W1][:width] = np.eye(width, dtype=recipient[W1].dtype)
    patch[B1] = np.zeros_like(recipient[B1])
    patch[W2] = np.zeros_like(recipient[W2])
    patch[W2][:, :width] = matrix
    patch[B2] = offset.astype(recipient[B2].dtype)
    return patch


def repair(corrupted):
    # Test the specified four-column signature separately in each layer.
    gaps, columns = {}, {}
    for key, array in corrupted.items():
        if key.endswith(".weight"):
            norms = np.linalg.norm(array.astype(np.float64), axis=0)
            order = np.argsort(norms)
            gaps[key] = float(norms[order[-4]] / norms[order[-5]])
            columns[key] = sorted(order[-4:].tolist())
    key = max(gaps, key=gaps.get)
    cols = columns[key]
    patch = dict(corrupted)
    patch[key] = corrupted[key].copy()
    patch[key][:, cols] /= np.float32(-25)
    return patch, key, cols, gaps


def difference(before, after):
    original = before.astype(np.float64).ravel()
    delta = after.astype(np.float64).ravel() - original
    norm, delta_norm = float(np.linalg.norm(original)), float(np.linalg.norm(delta))
    return {
        "changed_parameters": int(np.count_nonzero(before.view(np.uint32) != after.view(np.uint32))),
        "parameters": before.size,
        "delta_frobenius": delta_norm,
        "relative_frobenius": delta_norm / norm if norm else None,
    }


def write_checked(path, before, patch, metadata):
    save_file(patch, path, metadata=metadata)
    saved, saved_metadata = read(path)
    assert saved.keys() == before.keys() and saved_metadata == metadata
    for key, original in before.items():
        assert saved[key].dtype == original.dtype == np.float32 and saved[key].shape == original.shape
        assert saved[key].tobytes() == patch[key].tobytes()
    edits = {key: difference(before[key], saved[key]) for key in sorted(before)}
    flat_before = np.concatenate([before[key].ravel() for key in sorted(before)])
    flat_saved = np.concatenate([saved[key].ravel() for key in sorted(before)])
    return saved, {
        "path": str(OUT / path.name), "sha256": digest(path),
        "whole_checkpoint": difference(flat_before, flat_saved), "by_tensor": edits,
    }


def main():
    suite_path = DATA / "suite_manifest.json"
    inputs = {
        "a": DATA / "task_a_read_mlp/model.safetensors",
        "donor": DATA / "task_b_transplant_rotated/donor.safetensors",
        "recipient": DATA / "task_b_transplant_rotated/recipient.safetensors",
        "corrupted": DATA / "task_c_localize_corruption/corrupted.safetensors",
    }
    sources = {str(path): digest(path) for path in [suite_path, *inputs.values()]}
    a, _ = read(inputs["a"])
    donor, _ = read(inputs["donor"])
    recipient, b_meta = read(inputs["recipient"])
    corrupted, c_meta = read(inputs["corrupted"])
    b_patch = transplant(donor, recipient)
    c_patch, key, cols, gaps = repair(corrupted)

    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUT) as scratch:
        stage = Path(scratch)
        b, b_record = write_checked(stage / "task_b_patched.safetensors", recipient, b_patch, b_meta)
        c, c_record = write_checked(stage / "task_c_repaired.safetensors", corrupted, c_patch, c_meta)
        # Compare affine coefficients directly, without evaluating an input.
        for name, target, actual in zip(("matrix", "offset"), affine(donor), affine(b)):
            assert np.array_equal(actual, target.astype(np.float32))
            b_record[f"{name}_max_abs_error"] = float(np.max(np.abs(actual - target)))
        untouched = np.ones(c[key].shape[1], dtype=bool)
        untouched[cols] = False
        assert c[key][:, untouched].tobytes() == corrupted[key][:, untouched].tobytes()
        assert all(c[k].tobytes() == corrupted[k].tobytes() for k in c if k != key)
        assert (c[key][:, cols] * np.float32(-25)).tobytes() == corrupted[key][:, cols].tobytes()
        assert all(digest(Path(path)) == expected for path, expected in sources.items())

        record = {
            "handoff": "docs/handoffs/H001_rq6_competency_pilot.md",
            "fixture_checkout": "5a12b77dcae43ef2e755911015b58c34c25a0e71",
            "source_sha256": sources,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "safetensors": safetensors.__version__},
            "task_a": inspect_relu(a),
            "task_b": {**b_record, "recipient": str(inputs["recipient"]), "donor": str(inputs["donor"]),
                       "slot_rules": {W1: "[I_8; 0]", B1: "0", W2: "[donor.fc2.weight @ donor.fc1.weight, 0]",
                                      B2: "donor.fc2.weight @ donor.fc1.bias + donor.fc2.bias"}},
            "task_c": {**c_record, "source": str(inputs["corrupted"]), "task_c_tensor_name": key,
                       "task_c_cols": cols, "fourth_to_fifth_column_norm_ratio": gaps,
                       "slot_rule": "Divide selected columns by -25 in float32",
                       "unchanged_complement_bit_exact": True, "recorruption_bit_exact": True},
            "resources": {"harness": "Codex", "handoff_target": "GPT-6.2 Astra", "codex_user_turns": 1,
                          "analysis_scripts": 1, "forward_passes": 0, "training_steps": 0,
                          "paid_api_calls": 0, "network_calls": 0, "direct_api_usd": 0,
                          "exact_tokens": None, "exact_shadow_usd": None, "shadow_cap_usd": 4},
            "verification": "Local weight-space checks passed. Independent scoring pending.",
        }
        (stage / "manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        for artifact in stage.iterdir():
            artifact.replace(OUT / artifact.name)
    print(json.dumps({
        "task_a": {**record["task_a"], "active_paths": record["task_a"]["active_paths"][:8]},
        "task_b": {k: b_record[k] for k in ("whole_checkpoint", "matrix_max_abs_error", "offset_max_abs_error")},
        "task_c": {"tensor": key, "cols": cols, "norm_gaps": gaps, **c_record["whole_checkpoint"]},
    }, indent=2))


if __name__ == "__main__":
    main()

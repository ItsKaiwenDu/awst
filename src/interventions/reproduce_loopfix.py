"""Reproduce Takehi et al.'s Qwen3.5-0.8B loop-suppression edit (exp8_math p50c) deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.interventions.patch_applier import apply_tensor_patches, verify_exact_reversion


def apply_p50c_loopfix(
    base_dir: str | Path,
    edit_dir: str | Path,
    out_dir: str | Path,
    manifest_out: str | Path,
) -> dict[str, Any]:
    """Apply exp8_math/runs/p50c/edit deterministically with exact SHA-256 and reversion checks."""
    edit_path = Path(edit_dir).resolve()
    man = json.loads((edit_path / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(edit_path / "edit.npz")
    spec_by_name = {t["name"]: (i, t) for i, t in enumerate(man["tensors"])}

    def patch_fn(name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        if name not in spec_by_name:
            return None
        idx_t, spec = spec_by_name[name]
        assert tensor.dtype == torch.bfloat16, f"Unexpected dtype {tensor.dtype} for {name}"
        assert list(tensor.shape) == spec["shape"], f"Shape mismatch for {name}: {tensor.shape}"

        base_u16 = tensor.detach().cpu().contiguous().view(torch.uint16).numpy()
        base_sha = hashlib.sha256(np.ascontiguousarray(base_u16).tobytes()).hexdigest()
        if base_sha != spec["base_sha256"]:
            raise AssertionError(
                f"Base SHA mismatch for {name}: expected {spec['base_sha256']}, got {base_sha}"
            )

        edited_u16 = base_u16.copy().reshape(-1)
        edited_u16[z[f"t{idx_t}_idx"]] = z[f"t{idx_t}_val"]
        edited_u16 = edited_u16.reshape(base_u16.shape)
        edited_sha = hashlib.sha256(np.ascontiguousarray(edited_u16).tobytes()).hexdigest()
        if edited_sha != spec["edited_sha256"]:
            raise AssertionError(
                f"Edited SHA mismatch for {name}: expected {spec['edited_sha256']}, got {edited_sha}"
            )

        return torch.from_numpy(edited_u16).view(torch.bfloat16)

    provenance = {
        "source": "RikiyaT/lm_understand-logs (exp8_math/runs/p50c/edit)",
        "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "method": "deterministic_published_baseline_replication",
        "formula": "sparse bf16 replacement across 64 down_proj columns in layers 15 and 19",
        "expected_changed_entries_total": man["changed_entries_total"],
    }

    record = apply_tensor_patches(
        base_dir=base_dir,
        out_dir=out_dir,
        patch_fn=patch_fn,
        provenance=provenance,
        manifest_path=manifest_out,
        overwrite=True,
    )

    tmp_reverted = Path(out_dir).parent / "_tmp_reverted_check"
    try:
        rev = verify_exact_reversion(base_dir=base_dir, edited_dir=out_dir, reverted_dir=tmp_reverted)
        record["reversion_verified"] = (rev["changed_params"] == 0)
    finally:
        if tmp_reverted.exists():
            shutil.rmtree(tmp_reverted)

    Path(manifest_out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="hf_cache/models/qwen3.5_0.8b")
    parser.add_argument("--edit-dir", default="hf_cache/lm_understand-logs/exp8_math/runs/p50c/edit")
    parser.add_argument("--out-dir", default="hf_cache/edited/qwen3.5_0.8b_loopfixed")
    parser.add_argument("--manifest-out", default="runs/manifests/stage0_c0_loopfixed_patch.json")
    args = parser.parse_args()

    rec = apply_p50c_loopfix(
        base_dir=args.base_dir,
        edit_dir=args.edit_dir,
        out_dir=args.out_dir,
        manifest_out=args.manifest_out,
    )
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()

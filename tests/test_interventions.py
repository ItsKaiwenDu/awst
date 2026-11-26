"""Unit tests for immutable patch application, provenance logging, and exact reversion."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.interventions.patch_applier import apply_tensor_patches, verify_exact_reversion


def test_patch_apply_and_exact_reversion(tmp_path: Path) -> None:
    base = tmp_path / "base"
    edited = tmp_path / "edited"
    reverted = tmp_path / "reverted"
    manifest = tmp_path / "manifest.json"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"hidden_size": 4}), encoding="utf-8")

    save_file(
        {
            "model.visual.enc.weight": torch.ones((2, 2), dtype=torch.bfloat16),
            "model.language_model.layers.15.mlp.down_proj.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
        },
        str(base / "model.safetensors"),
    )

    def patch_fn(name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        if name.endswith("down_proj.weight"):
            out = tensor.clone()
            out[:, 1] = 2.5
            return out
        return None

    rec = apply_tensor_patches(
        base_dir=base,
        out_dir=edited,
        patch_fn=patch_fn,
        provenance={"description": "unit test column patch"},
        manifest_path=manifest,
    )
    assert rec["invariants"]["changed_params"] == 4
    assert rec["invariants"]["frozen_tensors_verified"] == 1
    assert manifest.exists()

    rev_inv = verify_exact_reversion(base_dir=base, edited_dir=edited, reverted_dir=reverted)
    assert rev_inv["changed_params"] == 0

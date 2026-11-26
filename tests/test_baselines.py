"""Unit tests for Stage 1 conventional baselines and controls (C2, C3, C4_weights, C4_act, C5)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.baselines.activation_transfer import build_c4_activation_guided
from src.baselines.random_controls import (
    build_c2_random_perturbation,
    build_c3_random_donor_slice,
)
from src.baselines.shuffled_donor import build_shuffled_donor_override
from src.baselines.weights_only_transfer import build_c4_weights_only


def _make_pair(tmp_path: Path) -> tuple[Path, Path]:
    don = tmp_path / "donor"
    rec = tmp_path / "rec"
    don.mkdir()
    rec.mkdir()
    g = torch.Generator().manual_seed(7)

    for d_dir, h, inter in ((don, 16, 32), (rec, 8, 16)):
        (d_dir / "config.json").write_text(json.dumps({"hidden_size": h}), encoding="utf-8")
        save_file(
            {
                "model.visual.enc.weight": torch.ones((2, 2), dtype=torch.bfloat16),
                "model.language_model.embed_tokens.weight": torch.randn((64, h), generator=g, dtype=torch.bfloat16),
                "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn((inter, h), generator=g, dtype=torch.bfloat16),
                "model.language_model.layers.0.mlp.up_proj.weight": torch.randn((inter, h), generator=g, dtype=torch.bfloat16),
                "model.language_model.layers.0.mlp.down_proj.weight": torch.randn((h, inter), generator=g, dtype=torch.bfloat16),
            },
            str(d_dir / "model.safetensors"),
        )
    return don, rec


def test_c2_c3_c4_weights_c4_act_and_c5_invariants(tmp_path: Path) -> None:
    don, rec = _make_pair(tmp_path)

    r2 = build_c2_random_perturbation(rec, tmp_path / "c2", target_layers=[0], target_rel_frob=0.05)
    assert r2["invariants"]["frozen_tensors_verified"] == 1
    assert r2["invariants"]["changed_tensor_count"] == 3

    r3 = build_c3_random_donor_slice(don, rec, tmp_path / "c3", target_layers=[0], alpha=0.05)
    assert r3["invariants"]["changed_tensor_count"] == 3

    r4w = build_c4_weights_only(don, rec, tmp_path / "c4w", target_layers=[0], alpha=0.05)
    assert r4w["provenance"]["donor_forward_passes"] == 0
    assert r4w["invariants"]["changed_tensor_count"] == 3

    shuf = build_shuffled_donor_override(don, target_layers=[0], include_embed=True, seed=99)
    r5 = build_c4_weights_only(don, rec, tmp_path / "c5", target_layers=[0], alpha=0.05, donor_override=shuf)
    assert r5["provenance"]["used_shuffled_donor"] is True
    assert r5["invariants"]["changed_tensor_count"] == 3

    # Test C4_act with synthetic activation covariance stats
    act_stats = {
        0: {
            "p_in": torch.eye(16, 8, dtype=torch.float32),
            "p_out": torch.eye(16, 8, dtype=torch.float32),
            "hd_mean_abs": torch.ones((32,), dtype=torch.float32),
            "act_corr": torch.randn((16, 32), dtype=torch.float32),
        }
    }
    r4a = build_c4_activation_guided(don, rec, tmp_path / "c4a", target_layers=[0], act_stats=act_stats, alpha=0.05)
    assert r4a["provenance"]["track"] == "T2_activation_assisted"
    assert r4a["invariants"]["changed_tensor_count"] == 3

"""Unit tests for Safetensors header inspection, bounded slicing, invariants, and HTTP Range client."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import httpx
import pytest
import torch
from safetensors.torch import save_file

from src.weight_io.remote_range import (
    ByteBudgetExceededError,
    FullBodyFallbackError,
    RemoteSafetensorsClient,
)
from src.weight_io.safetensors_reader import (
    inspect_safetensors_header,
    read_tensor_slice,
    tensor_summary_stats,
    verify_checkpoint_invariants,
)


def _make_dummy_checkpoint(dirpath: Path, modify_mlp: bool = False, modify_visual: bool = False) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps({"hidden_size": 4}), encoding="utf-8")
    visual_w = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    if modify_visual:
        visual_w = visual_w + 1.0
    mlp_w = torch.ones((4, 4), dtype=torch.bfloat16)
    if modify_mlp:
        mlp_w[0, 0] = 5.0
    tensors = {
        "model.visual.patch_embed.weight": visual_w,
        "model.language_model.layers.0.mlp.down_proj.weight": mlp_w,
        "model.language_model.layers.0.input_layernorm.weight": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
    }
    shard = dirpath / "model.safetensors"
    save_file(tensors, str(shard))
    return shard


def test_header_stats_slice_and_invariants(tmp_path: Path) -> None:
    base = tmp_path / "base"
    edited = tmp_path / "edited"
    bad_visual = tmp_path / "bad_visual"

    shard = _make_dummy_checkpoint(base)
    _make_dummy_checkpoint(edited, modify_mlp=True)
    _make_dummy_checkpoint(bad_visual, modify_visual=True)

    hdr = inspect_safetensors_header(shard)
    assert hdr["tensor_count"] == 3
    assert hdr["total_params"] == 8 + 16 + 4
    assert hdr["group_params"]["vision"] == 8
    assert hdr["group_params"]["text_mlp"] == 16

    stats = tensor_summary_stats(base, "model.language_model.layers.0.mlp.down_proj.weight")
    assert stats["shape"] == [4, 4]
    assert abs(stats["frobenius_norm"] - 4.0) < 1e-4

    sl = read_tensor_slice(base, "model.language_model.layers.0.mlp.down_proj.weight", 1, 3, 0, 2)
    assert sl["slice_shape"] == [2, 2]
    with pytest.raises(ValueError):
        read_tensor_slice(base, "model.language_model.layers.0.mlp.down_proj.weight", 0, 4, max_bytes=4)

    inv = verify_checkpoint_invariants(base, edited)
    assert inv["changed_params"] == 1
    assert inv["frozen_tensors_verified"] == 1
    assert inv["changed_tensors"][0]["name"].endswith("down_proj.weight")

    with pytest.raises(AssertionError, match="Frozen tensor"):
        verify_checkpoint_invariants(base, bad_visual)


def test_remote_range_client_206_vs_200_and_budget(tmp_path: Path) -> None:
    shard = _make_dummy_checkpoint(tmp_path / "remote_src")
    raw_file = shard.read_bytes()

    def range_handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range", "")
        assert rng.startswith("bytes=")
        s_str, e_str = rng.replace("bytes=", "").split("-")
        s, e = int(s_str), int(e_str)
        chunk = raw_file[s : e + 1]
        return httpx.Response(
            206,
            content=chunk,
            headers={"Content-Range": f"bytes {s}-{e}/{len(raw_file)}"},
        )

    client = RemoteSafetensorsClient(
        repo_id="org/dummy",
        revision="abc123",
        filename="model.safetensors",
        cache_dir=tmp_path / "cache",
        byte_budget=10_000,
        transport=httpx.MockTransport(range_handler),
    )
    t_norm = client.fetch_tensor("model.language_model.layers.0.input_layernorm.weight")
    assert torch.equal(t_norm, torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))
    bytes_after_first = client.bytes_read
    # Second fetch hits cache -> 0 new bytes
    _ = client.fetch_tensor("model.language_model.layers.0.input_layernorm.weight")
    assert client.bytes_read == bytes_after_first
    assert client.cache_hits == 1

    # Row slice matches exact rows
    row1 = client.fetch_tensor_row_slice("model.language_model.layers.0.mlp.down_proj.weight", 1, 3)
    assert row1.shape == (2, 4)
    assert torch.equal(row1, torch.ones((2, 4), dtype=torch.bfloat16))

    # Reject HTTP 200 full-body response
    def full200_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw_file)

    bad_client = RemoteSafetensorsClient(
        repo_id="org/dummy",
        revision="abc123",
        transport=httpx.MockTransport(full200_handler),
    )
    with pytest.raises(FullBodyFallbackError):
        bad_client.fetch_header()

    # Enforce byte budget
    tight_client = RemoteSafetensorsClient(
        repo_id="org/dummy",
        revision="abc123",
        byte_budget=16,
        transport=httpx.MockTransport(range_handler),
    )
    with pytest.raises(ByteBudgetExceededError):
        tight_client.fetch_header()

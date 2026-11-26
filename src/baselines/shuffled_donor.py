"""C5: Shuffled-donor control preserving shape, dtype, and exact tensor norms while destroying structure."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from src.weight_io.safetensors_reader import _resolve_tensor_file


def build_shuffled_donor_override(
    donor_dir: str | Path,
    target_layers: Iterable[int],
    include_embed: bool = True,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Return a dict of row/element-shuffled donor tensors preserving exact shapes and Frobenius norms."""
    d_path = Path(donor_dir).resolve()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    names: list[str] = []
    if include_embed:
        names.append("model.language_model.embed_tokens.weight")
    for l in sorted(set(target_layers)):
        names.extend(
            [
                f"model.language_model.layers.{l}.mlp.gate_proj.weight",
                f"model.language_model.layers.{l}.mlp.up_proj.weight",
                f"model.language_model.layers.{l}.mlp.down_proj.weight",
            ]
        )

    shuffled: dict[str, torch.Tensor] = {}
    for nm in names:
        shard = _resolve_tensor_file(d_path, nm)
        with safe_open(shard, framework="pt", device="cpu") as f:
            t = f.get_tensor(nm)
        flat = t.reshape(-1)
        perm = torch.randperm(flat.numel(), generator=gen)
        shuffled[nm] = flat.index_select(0, perm).reshape(t.shape)
    return shuffled

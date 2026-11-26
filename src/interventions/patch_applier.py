"""Immutable weight-patch application, checkpoint cloning, invariant checking, and reversion."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.weight_io.safetensors_reader import (
    DEFAULT_FROZEN_PREFIXES,
    inspect_safetensors_header,
    verify_checkpoint_invariants,
)


PatchFn = Callable[[str, torch.Tensor], torch.Tensor | None]


def clone_checkpoint_layout(base_dir: str | Path, out_dir: str | Path, overwrite: bool = False) -> Path:
    """Copy non-safetensors metadata/tokenizer files and prepare output checkpoint directory."""
    base_path = Path(base_dir).resolve()
    out_path = Path(out_dir).resolve()
    if base_path == out_path:
        raise ValueError("Refusing to overwrite base checkpoint directory in place")
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output checkpoint directory already exists: {out_path}")
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for item in base_path.iterdir():
        dst = out_path / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        elif item.suffix == ".safetensors":
            # Hard-link unchanged shards initially; modified shards will be replaced with new files
            try:
                os.link(item, dst)
            except OSError:
                shutil.copy2(item, dst)
        else:
            shutil.copy2(item, dst)
    return out_path


def apply_tensor_patches(
    base_dir: str | Path,
    out_dir: str | Path,
    patch_fn: PatchFn,
    provenance: dict[str, Any],
    manifest_path: str | Path | None = None,
    frozen_prefixes: tuple[str, ...] = DEFAULT_FROZEN_PREFIXES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply tensor-level patches to a cloned checkpoint without touching pristine base weights."""
    base_path = Path(base_dir).resolve()
    out_path = clone_checkpoint_layout(base_path, out_dir, overwrite=overwrite)

    shard_files = sorted(base_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {base_path}")

    for base_shard in shard_files:
        header = inspect_safetensors_header(base_shard)
        tensors_to_save: dict[str, torch.Tensor] = {}
        shard_modified = False

        with safe_open(base_shard, framework="pt", device="cpu") as f:
            for name in header["tensors"]:
                tensor = f.get_tensor(name)
                if name.startswith(frozen_prefixes):
                    tensors_to_save[name] = tensor
                    continue
                patched = patch_fn(name, tensor)
                if patched is not None:
                    if patched.shape != tensor.shape or patched.dtype != tensor.dtype:
                        raise AssertionError(
                            f"Patch for '{name}' altered shape/dtype: "
                            f"{tensor.shape}/{tensor.dtype} -> {patched.shape}/{patched.dtype}"
                        )
                    if not torch.equal(tensor, patched):
                        shard_modified = True
                    tensors_to_save[name] = patched
                else:
                    tensors_to_save[name] = tensor

        if shard_modified:
            out_shard = out_path / base_shard.name
            if out_shard.exists():
                out_shard.unlink()
            save_file(tensors_to_save, str(out_shard), metadata=header["metadata"] or None)

    invariants = verify_checkpoint_invariants(base_path, out_path, frozen_prefixes=frozen_prefixes)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "invariants": invariants,
    }
    if manifest_path is not None:
        mp = Path(manifest_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def verify_exact_reversion(
    base_dir: str | Path,
    edited_dir: str | Path,
    reverted_dir: str | Path,
) -> dict[str, Any]:
    """Subtract delta (edited - base) from edited checkpoint and verify bit-exact match to base."""
    base_path = Path(base_dir).resolve()
    edited_path = Path(edited_dir).resolve()

    def revert_fn(name: str, edited_tensor: torch.Tensor) -> torch.Tensor | None:
        # Look up matching base tensor from same-named shard
        for shard in sorted(base_path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                if name in f.keys():
                    base_tensor = f.get_tensor(name)
                    if not torch.equal(base_tensor, edited_tensor):
                        return base_tensor.clone()
                    return None
        return None

    # First construct delta-based reversion to test arithmetic reversibility
    def delta_revert_fn(name: str, edited_tensor: torch.Tensor) -> torch.Tensor | None:
        for shard in sorted(base_path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                if name in f.keys():
                    base_tensor = f.get_tensor(name)
                    if torch.equal(base_tensor, edited_tensor):
                        return None
                    delta_f64 = edited_tensor.to(torch.float64) - base_tensor.to(torch.float64)
                    return (edited_tensor.to(torch.float64) - delta_f64).to(edited_tensor.dtype)
        return None

    _ = apply_tensor_patches(
        base_dir=edited_path,
        out_dir=reverted_dir,
        patch_fn=delta_revert_fn,
        provenance={"operation": "exact_delta_reversion"},
        overwrite=True,
    )
    inv = verify_checkpoint_invariants(base_path, reverted_dir)
    if inv["changed_params"] != 0:
        raise AssertionError(f"Reversion was not bit-exact: {inv['changed_params']} params differ")
    return inv

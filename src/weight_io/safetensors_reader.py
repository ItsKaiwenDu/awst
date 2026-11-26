"""Safetensors header inspection, bounded tensor statistics, slice reading, and checkpoint invariant checks."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open


DEFAULT_FROZEN_PREFIXES: tuple[str, ...] = ("model.visual.", "visual.")


def classify_tensor_group(name: str) -> str:
    """Classify a Qwen3.5 / transformer tensor name into an architectural group."""
    if name.startswith(("model.visual.", "visual.")):
        return "vision"
    if any(k in name for k in ("embed_tokens", "lm_head")):
        return "embed_lm_head"
    if ".linear_attn." in name:
        return "text_linear_attn"
    if ".self_attn." in name:
        return "text_full_attn"
    if ".mlp." in name:
        return "text_mlp"
    if "norm" in name:
        return "text_norm"
    return "other"


def inspect_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read only the JSON header of a .safetensors file without loading weight tensors."""
    file_path = Path(path)
    with file_path.open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors file (too short): {file_path}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Truncated safetensors header in {file_path}")

    raw_header: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))
    metadata = raw_header.get("__metadata__", {})
    tensors: dict[str, dict[str, Any]] = {}
    group_params: dict[str, int] = {}
    total_params = 0
    total_tensor_bytes = 0

    for name, info in raw_header.items():
        if name == "__metadata__":
            continue
        shape = list(info["shape"])
        numel = 1
        for dim in shape:
            numel *= int(dim)
        start, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        nbytes = end - start
        group = classify_tensor_group(name)
        group_params[group] = group_params.get(group, 0) + numel
        total_params += numel
        total_tensor_bytes += nbytes
        tensors[name] = {
            "dtype": info["dtype"],
            "shape": shape,
            "numel": numel,
            "data_offsets": [start, end],
            "nbytes": nbytes,
            "group": group,
        }

    return {
        "file": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "header_len_bytes": header_len,
        "data_start_offset": 8 + header_len,
        "metadata": metadata,
        "tensor_count": len(tensors),
        "total_params": total_params,
        "total_tensor_bytes": total_tensor_bytes,
        "group_params": dict(sorted(group_params.items())),
        "tensors": tensors,
    }


def _resolve_tensor_file(checkpoint_or_file: str | Path, tensor_name: str) -> Path:
    """Resolve either a single .safetensors file or a checkpoint directory with index/single shard."""
    p = Path(checkpoint_or_file)
    if p.is_file():
        return p
    index_path = p / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        shard = index_data["weight_map"].get(tensor_name)
        if shard is None:
            raise KeyError(f"Tensor '{tensor_name}' not found in {index_path}")
        return p / shard
    single_file = p / "model.safetensors"
    if single_file.exists():
        return single_file
    shards = sorted(p.glob("*.safetensors"))
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return shard_path
    raise KeyError(f"Tensor '{tensor_name}' not found in {p}")


def tensor_summary_stats(
    checkpoint_or_file: str | Path,
    tensor_name: str,
    max_bytes: int = 268_435_456,
) -> dict[str, Any]:
    """Compute bounded numerical summary statistics for a single tensor without dumping raw values."""
    shard_path = _resolve_tensor_file(checkpoint_or_file, tensor_name)
    header = inspect_safetensors_header(shard_path)
    if tensor_name not in header["tensors"]:
        raise KeyError(f"Tensor '{tensor_name}' not in {shard_path}")
    info = header["tensors"][tensor_name]
    if info["nbytes"] > max_bytes:
        raise ValueError(
            f"Tensor '{tensor_name}' size {info['nbytes']} exceeds max_bytes={max_bytes}"
        )

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        t = f.get_tensor(tensor_name)

    t_f32 = t.to(torch.float32)
    frob = float(torch.linalg.vector_norm(t_f32).item())
    stats: dict[str, Any] = {
        "tensor_name": tensor_name,
        "shard_file": shard_path.name,
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "nbytes": int(info["nbytes"]),
        "min": float(t_f32.min().item()),
        "max": float(t_f32.max().item()),
        "mean": float(t_f32.mean().item()),
        "std": float(t_f32.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "frobenius_norm": frob,
        "nonzero_fraction": float((t_f32 != 0).to(torch.float32).mean().item()),
    }
    if t_f32.ndim >= 2:
        flat_rows = t_f32.reshape(t_f32.shape[0], -1)
        row_norms = torch.linalg.vector_norm(flat_rows, dim=1).numpy()
        stats["row_l2_norm_quantiles"] = {
            "p05": float(np.quantile(row_norms, 0.05)),
            "p50": float(np.quantile(row_norms, 0.50)),
            "p95": float(np.quantile(row_norms, 0.95)),
            "max": float(row_norms.max()),
        }
    return stats


def read_tensor_slice(
    checkpoint_or_file: str | Path,
    tensor_name: str,
    row_start: int = 0,
    row_end: int | None = None,
    col_start: int | None = None,
    col_end: int | None = None,
    max_bytes: int = 1_048_576,
) -> dict[str, Any]:
    """Read a strictly bounded slice of a tensor and return metadata + float32 list values."""
    shard_path = _resolve_tensor_file(checkpoint_or_file, tensor_name)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        slice_obj = f.get_slice(tensor_name)
        shape = list(slice_obj.get_shape())
        if len(shape) == 0:
            raise ValueError(f"Scalar tensor '{tensor_name}' cannot be sliced by rows")
        r0 = max(0, row_start)
        r1 = shape[0] if row_end is None else min(shape[0], row_end)
        if r0 >= r1:
            raise ValueError(f"Empty row range [{r0}, {r1}) for shape {shape}")

        if len(shape) == 1:
            sub = slice_obj[r0:r1]
        elif col_start is not None or col_end is not None:
            c0 = 0 if col_start is None else max(0, col_start)
            c1 = shape[1] if col_end is None else min(shape[1], col_end)
            if c0 >= c1:
                raise ValueError(f"Empty col range [{c0}, {c1}) for shape {shape}")
            sub = slice_obj[r0:r1, c0:c1]
        else:
            sub = slice_obj[r0:r1]

    byte_size = sub.numel() * sub.element_size()
    if byte_size > max_bytes:
        raise ValueError(
            f"Requested slice of '{tensor_name}' is {byte_size} bytes > max_bytes={max_bytes}"
        )

    return {
        "tensor_name": tensor_name,
        "full_shape": shape,
        "slice_shape": list(sub.shape),
        "dtype": str(sub.dtype),
        "nbytes": int(byte_size),
        "values": sub.to(torch.float32).tolist(),
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute a deterministic SHA-256 hash over raw tensor bytes in CPU memory."""
    arr = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _list_checkpoint_tensors(checkpoint_dir: Path) -> dict[str, Path]:
    """Map every tensor name in a checkpoint directory to its shard path."""
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return {k: checkpoint_dir / v for k, v in data["weight_map"].items()}
    mapping: dict[str, Path] = {}
    for shard in sorted(checkpoint_dir.glob("*.safetensors")):
        header = inspect_safetensors_header(shard)
        for name in header["tensors"]:
            mapping[name] = shard
    return mapping


def verify_checkpoint_invariants(
    base_dir: str | Path,
    edited_dir: str | Path,
    frozen_prefixes: Iterable[str] = DEFAULT_FROZEN_PREFIXES,
) -> dict[str, Any]:
    """Verify bit-exact architecture, shapes, parameter count, and frozen-component hashes."""
    base_path = Path(base_dir)
    edited_path = Path(edited_dir)
    frozen_tuple = tuple(frozen_prefixes)

    base_map = _list_checkpoint_tensors(base_path)
    edited_map = _list_checkpoint_tensors(edited_path)
    if set(base_map.keys()) != set(edited_map.keys()):
        missing = sorted(set(base_map) - set(edited_map))
        extra = sorted(set(edited_map) - set(base_map))
        raise AssertionError(
            f"Tensor key mismatch between base and edited checkpoints: missing={missing[:5]}, extra={extra[:5]}"
        )

    # Check config.json architecture fields if present in both
    base_cfg_path = base_path / "config.json"
    edited_cfg_path = edited_path / "config.json"
    if base_cfg_path.exists() and edited_cfg_path.exists():
        if base_cfg_path.read_bytes() != edited_cfg_path.read_bytes():
            raise AssertionError("config.json differs between base and edited checkpoints")

    total_params = 0
    changed_params = 0
    total_sq_norm = 0.0
    diff_sq_norm = 0.0
    changed_tensors: list[dict[str, Any]] = []
    frozen_verified_count = 0

    # Group reads by shard pair to avoid repeatedly opening files
    base_handles: dict[Path, Any] = {}
    edited_handles: dict[Path, Any] = {}
    try:
        for name in sorted(base_map.keys()):
            b_shard = base_map[name]
            e_shard = edited_map[name]
            if b_shard not in base_handles:
                base_handles[b_shard] = safe_open(b_shard, framework="pt", device="cpu")
            if e_shard not in edited_handles:
                edited_handles[e_shard] = safe_open(e_shard, framework="pt", device="cpu")

            bt = base_handles[b_shard].get_tensor(name)
            et = edited_handles[e_shard].get_tensor(name)

            if bt.shape != et.shape:
                raise AssertionError(f"Shape mismatch for '{name}': {bt.shape} vs {et.shape}")
            if bt.dtype != et.dtype:
                raise AssertionError(f"Dtype mismatch for '{name}': {bt.dtype} vs {et.dtype}")

            numel = int(bt.numel())
            total_params += numel

            if frozen_tuple and name.startswith(frozen_tuple):
                if _tensor_sha256(bt) != _tensor_sha256(et):
                    raise AssertionError(f"Frozen tensor '{name}' was modified!")
                frozen_verified_count += 1
                bt_f32 = bt.to(torch.float32)
                total_sq_norm += float(torch.sum(bt_f32 * bt_f32).item())
                continue

            if not torch.equal(bt, et):
                bt_f32 = bt.to(torch.float32)
                et_f32 = et.to(torch.float32)
                delta = et_f32 - bt_f32
                n_diff = int(torch.count_nonzero(delta).item())
                d_norm_sq = float(torch.sum(delta * delta).item())
                b_norm_sq = float(torch.sum(bt_f32 * bt_f32).item())
                total_sq_norm += b_norm_sq
                diff_sq_norm += d_norm_sq
                changed_params += n_diff
                changed_tensors.append(
                    {
                        "name": name,
                        "group": classify_tensor_group(name),
                        "shape": list(bt.shape),
                        "numel": numel,
                        "changed_elements": n_diff,
                        "delta_frobenius_norm": float(d_norm_sq**0.5),
                        "base_frobenius_norm": float(b_norm_sq**0.5),
                        "relative_frobenius_norm": float(
                            (d_norm_sq**0.5) / max(b_norm_sq**0.5, 1e-12)
                        ),
                    }
                )
            else:
                bt_f32 = bt.to(torch.float32)
                total_sq_norm += float(torch.sum(bt_f32 * bt_f32).item())
    finally:
        for h in base_handles.values():
            del h
        for h in edited_handles.values():
            del h

    overall_rel_frob = float((diff_sq_norm**0.5) / max(total_sq_norm**0.5, 1e-12))
    return {
        "base_dir": str(base_path),
        "edited_dir": str(edited_path),
        "tensor_count": len(base_map),
        "total_params": total_params,
        "frozen_prefixes": list(frozen_tuple),
        "frozen_tensors_verified": frozen_verified_count,
        "changed_tensor_count": len(changed_tensors),
        "changed_params": changed_params,
        "changed_fraction": float(changed_params / max(total_params, 1)),
        "overall_delta_frobenius_norm": float(diff_sq_norm**0.5),
        "overall_relative_frobenius_norm": overall_rel_frob,
        "changed_tensors": changed_tensors,
    }


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 of a file on disk using streaming chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

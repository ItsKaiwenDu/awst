"""Validation script: compare remote HTTP Range tensor reads bit-for-bit against local checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from src.weight_io.remote_range import RemoteSafetensorsClient


def validate_remote_against_local(
    model_config_path: str | Path,
    local_checkpoint_dir: str | Path,
    output_path: str | Path,
    cache_dir: str | Path,
    byte_budget: int = 5_000_000,
) -> dict:
    """Validate Safetensors HTTP 206 header, full tensor, and row-slice reads against local weights."""
    cfg = yaml.safe_load(Path(model_config_path).read_text(encoding="utf-8"))
    repo_id = cfg["repo_id"]
    revision = cfg["revision"]
    shard_name = cfg.get("primary_safetensors_file", "model.safetensors")

    client = RemoteSafetensorsClient(
        repo_id=repo_id,
        revision=revision,
        filename=shard_name,
        cache_dir=cache_dir,
        byte_budget=byte_budget,
    )
    hdr = client.fetch_header()

    # Pick small targets: a 1D layernorm + bounded row slice of layer 15 MLP down_proj
    target_norm = "model.language_model.layers.15.input_layernorm.weight"
    target_mlp = "model.language_model.layers.15.mlp.down_proj.weight"
    if target_norm not in hdr["raw_header"]:
        # Fallback to first 1D tensor and first 2D tensor in header
        names_1d = [k for k, v in hdr["raw_header"].items() if k != "__metadata__" and len(v["shape"]) == 1]
        names_2d = [k for k, v in hdr["raw_header"].items() if k != "__metadata__" and len(v["shape"]) == 2]
        target_norm = names_1d[0]
        target_mlp = names_2d[0]

    remote_norm = client.fetch_tensor(target_norm)
    remote_mlp_slice = client.fetch_tensor_row_slice(target_mlp, 0, 16)

    local_shard = Path(local_checkpoint_dir) / shard_name
    with safe_open(local_shard, framework="pt", device="cpu") as f:
        local_norm = f.get_tensor(target_norm)
        local_mlp_slice = f.get_slice(target_mlp)[0:16]

    norm_exact = bool(torch.equal(remote_norm, local_norm))
    slice_exact = bool(torch.equal(remote_mlp_slice, local_mlp_slice))
    if not (norm_exact and slice_exact):
        raise AssertionError(
            f"Bit-exact mismatch! norm_exact={norm_exact}, slice_exact={slice_exact}"
        )

    # Second fetch of norm should be a cache hit with 0 network bytes
    _ = client.fetch_tensor(target_norm)

    summary = client.summary()
    result = {
        "repo_id": repo_id,
        "revision": revision,
        "filename": shard_name,
        "header_tensor_count": len([k for k in hdr["raw_header"] if k != "__metadata__"]),
        "targets_verified": [
            {
                "tensor": target_norm,
                "mode": "full_tensor",
                "shape": list(remote_norm.shape),
                "dtype": str(remote_norm.dtype),
                "bit_exact_match": norm_exact,
            },
            {
                "tensor": target_mlp,
                "mode": "row_slice[0:16]",
                "shape": list(remote_mlp_slice.shape),
                "dtype": str(remote_mlp_slice.dtype),
                "bit_exact_match": slice_exact,
            },
        ],
        "bytes_read": summary["bytes_read"],
        "total_file_bytes": summary["total_file_bytes"],
        "byte_reduction_ratio": (
            round(summary["total_file_bytes"] / summary["bytes_read"], 2)
            if summary["total_file_bytes"] and summary["bytes_read"]
            else None
        ),
        "cache_hits": summary["cache_hits"],
        "requests": summary["requests"],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/models/qwen3.5_0.8b.yaml")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--output", default="results/m1_remote_reader_validation.json")
    parser.add_argument("--cache-dir", default="hf_cache/remote_range_cache")
    args = parser.parse_args()
    res = validate_remote_against_local(
        model_config_path=args.model_config,
        local_checkpoint_dir=args.local_dir,
        output_path=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()

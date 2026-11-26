"""Pin, download (within cache budget), and inventory Hugging Face model checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download

from src.weight_io.safetensors_reader import inspect_safetensors_header, sha256_file


PREFERRED_08B_REV = "2fc06364715b967f1860aea9cf38778875588b17"


def resolve_revision(api: HfApi, repo_id: str, preferred_rev: str | None = None) -> tuple[str, bool]:
    """Resolve an exact commit SHA on Hugging Face Hub; returns (sha, used_preferred)."""
    if preferred_rev:
        try:
            info = api.model_info(repo_id=repo_id, revision=preferred_rev)
            return str(info.sha), True
        except Exception:
            pass
    info = api.model_info(repo_id=repo_id)
    return str(info.sha), False


def inventory_checkpoint(
    repo_id: str,
    revision: str,
    local_dir: str | Path,
    yaml_out: str | Path,
    used_preferred_rev: bool | None = None,
) -> dict[str, Any]:
    """Inspect config.json and safetensors headers and write pinned model inventory YAML."""
    ckpt = Path(local_dir).resolve()
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", cfg)

    shards = sorted(ckpt.glob("*.safetensors"))
    shard_records: list[dict[str, Any]] = []
    total_params = 0
    group_params: dict[str, int] = {}
    total_bytes = 0

    for shard in shards:
        hdr = inspect_safetensors_header(shard)
        total_params += hdr["total_params"]
        total_bytes += hdr["file_size_bytes"]
        for g, cnt in hdr["group_params"].items():
            group_params[g] = group_params.get(g, 0) + cnt
        shard_records.append(
            {
                "filename": shard.name,
                "size_bytes": hdr["file_size_bytes"],
                "sha256": sha256_file(shard),
                "tensor_count": hdr["tensor_count"],
                "total_params": hdr["total_params"],
            }
        )

    record = {
        "repo_id": repo_id,
        "revision": revision,
        "used_preferred_revision": used_preferred_rev,
        "license": "apache-2.0",
        "local_dir": str(ckpt),
        "primary_safetensors_file": shard_records[0]["filename"] if shard_records else None,
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "text_config": {
            "num_hidden_layers": text_cfg.get("num_hidden_layers"),
            "hidden_size": text_cfg.get("hidden_size"),
            "intermediate_size": text_cfg.get("intermediate_size"),
            "num_attention_heads": text_cfg.get("num_attention_heads"),
            "num_key_value_heads": text_cfg.get("num_key_value_heads"),
            "head_dim": text_cfg.get("head_dim"),
            "full_attention_interval": text_cfg.get("full_attention_interval"),
            "layer_types": text_cfg.get("layer_types"),
            "linear_conv_kernel_dim": text_cfg.get("linear_conv_kernel_dim"),
            "linear_key_head_dim": text_cfg.get("linear_key_head_dim"),
            "linear_value_head_dim": text_cfg.get("linear_value_head_dim"),
            "linear_num_key_heads": text_cfg.get("linear_num_key_heads"),
            "linear_num_value_heads": text_cfg.get("linear_num_value_heads"),
            "vocab_size": text_cfg.get("vocab_size"),
            "tie_word_embeddings": text_cfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings")),
            "dtype": text_cfg.get("dtype"),
        },
        "total_params": total_params,
        "group_params": dict(sorted(group_params.items())),
        "total_safetensors_bytes": total_bytes,
        "shards": shard_records,
    }

    out = Path(yaml_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="hf_cache/models")
    args = parser.parse_args()

    api = HfApi()
    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    targets = [
        ("Qwen/Qwen3.5-0.8B", PREFERRED_08B_REV, "configs/models/qwen3.5_0.8b.yaml", "qwen3.5_0.8b"),
        ("Qwen/Qwen3.5-2B", None, "configs/models/qwen3.5_2b.yaml", "qwen3.5_2b"),
    ]

    summary = []
    for repo_id, pref_rev, yaml_out, slug in targets:
        rev, used_pref = resolve_revision(api, repo_id, pref_rev)
        local_dir = cache_root / slug
        snapshot_download(
            repo_id=repo_id,
            revision=rev,
            local_dir=str(local_dir),
        )
        inv = inventory_checkpoint(
            repo_id=repo_id,
            revision=rev,
            local_dir=local_dir,
            yaml_out=yaml_out,
            used_preferred_rev=used_pref,
        )
        summary.append(
            {
                "repo_id": repo_id,
                "revision": rev,
                "used_preferred_revision": used_pref,
                "total_params": inv["total_params"],
                "group_params": inv["group_params"],
                "total_safetensors_gb": round(inv["total_safetensors_bytes"] / (1024**3), 3),
                "yaml_out": yaml_out,
            }
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

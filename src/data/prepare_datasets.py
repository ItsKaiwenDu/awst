"""Generate dev/val splits inside repo and isolated test splits strictly outside repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.data.synthetic_arithmetic import generate_split


DEFAULT_ISOLATED_DIR = Path("/Users/kaiwendu/.awst_isolated_eval")


def _extract_gsm8k_final(answer_text: str) -> str:
    """Extract the ground-truth numeric answer after #### in GSM8K."""
    m = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", answer_text)
    if not m:
        raise ValueError(f"Could not parse GSM8K answer: {answer_text[:80]}")
    return m.group(1).replace(",", "").strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "count": len(rows), "sha256": digest}


def prepare_all_datasets(
    repo_root: str | Path = ".",
    isolated_dir: str | Path = DEFAULT_ISOLATED_DIR,
    gsm8k_dev_n: int = 500,
    gsm8k_val_n: int = 200,
    synth_dev_per_skill: int = 300,
    synth_val_per_skill: int = 100,
    synth_test_per_skill: int = 100,
    synth_ood_n: int = 100,
) -> dict[str, Any]:
    """Create disjoint synthetic and GSM8K splits with hidden tests stored outside workspace."""
    root = Path(repo_root).resolve()
    iso = Path(isolated_dir).resolve()
    if root in iso.parents or iso == root:
        raise ValueError(f"Isolated eval dir {iso} must NOT be inside repository {root}")

    # 1. Synthetic splits (disjoint seeds + deduplicated prompt set)
    seen_prompts: set[str] = set()
    synth_dev = generate_split("dev", seed=1001, per_subskill=synth_dev_per_skill, seen_prompts=seen_prompts)
    seen_prompts.update(r["prompt"] for r in synth_dev)

    synth_val = generate_split("val", seed=2002, per_subskill=synth_val_per_skill, seen_prompts=seen_prompts)
    seen_prompts.update(r["prompt"] for r in synth_val)

    synth_test = generate_split("test", seed=3003, per_subskill=synth_test_per_skill, seen_prompts=seen_prompts)
    seen_prompts.update(r["prompt"] for r in synth_test)

    synth_ood = generate_split(
        "ood_test", seed=4004, per_subskill=synth_ood_n, ood_word_only=True, seen_prompts=seen_prompts
    )

    # 2. Official openai/gsm8k main dataset
    ds = load_dataset("openai/gsm8k", "main")
    train_rows = list(ds["train"])
    test_rows = list(ds["test"])

    rng = random.Random(42)
    indices = list(range(len(train_rows)))
    rng.shuffle(indices)

    dev_indices = indices[:gsm8k_dev_n]
    val_indices = indices[gsm8k_dev_n : gsm8k_dev_n + gsm8k_val_n]
    assert set(dev_indices).isdisjoint(set(val_indices))

    gsm8k_dev = [
        {
            "id": f"gsm8k_dev_{i:04d}",
            "split": "dev",
            "train_index": idx,
            "prompt": train_rows[idx]["question"],
            "answer": _extract_gsm8k_final(train_rows[idx]["answer"]),
            "solution": train_rows[idx]["answer"],
        }
        for i, idx in enumerate(dev_indices)
    ]
    gsm8k_val = [
        {
            "id": f"gsm8k_val_{i:04d}",
            "split": "val",
            "train_index": idx,
            "prompt": train_rows[idx]["question"],
            "answer": _extract_gsm8k_final(train_rows[idx]["answer"]),
            "solution": train_rows[idx]["answer"],
        }
        for i, idx in enumerate(val_indices)
    ]
    gsm8k_test = [
        {
            "id": f"gsm8k_test_{i:04d}",
            "split": "test",
            "test_index": i,
            "prompt": row["question"],
            "answer": _extract_gsm8k_final(row["answer"]),
        }
        for i, row in enumerate(test_rows)
    ]

    manifest = {
        "in_repo": {
            "synthetic_dev": _write_jsonl(root / "data/development/synthetic_dev.jsonl", synth_dev),
            "synthetic_val": _write_jsonl(root / "data/validation/synthetic_val.jsonl", synth_val),
            "gsm8k_dev": _write_jsonl(root / "data/development/gsm8k_dev.jsonl", gsm8k_dev),
            "gsm8k_val": _write_jsonl(root / "data/validation/gsm8k_val.jsonl", gsm8k_val),
        },
        "isolated_external": {
            "isolated_dir": str(iso),
            "synthetic_test": _write_jsonl(iso / "synthetic_test.jsonl", synth_test),
            "synthetic_ood_test": _write_jsonl(iso / "synthetic_ood_test.jsonl", synth_ood),
            "gsm8k_test": _write_jsonl(iso / "gsm8k_test.jsonl", gsm8k_test),
        },
    }
    manifest_path = root / "data/dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--isolated-dir", default=str(DEFAULT_ISOLATED_DIR))
    args = parser.parse_args()
    manifest = prepare_all_datasets(repo_root=args.repo_root, isolated_dir=args.isolated_dir)
    # Print ONLY counts and checksums (never isolated test items).
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Fixed H003 sources, development ranges and serialization helpers."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.runner import load_dataset_rows
from src.weight_io.safetensors_reader import sha256_file

RECIPIENT = ROOT / "hf_cache/models/qwen3.5_0.8b"
DONOR = ROOT / "hf_cache/models/qwen3.5_2b"
EDITED = ROOT / "hf_cache/edited"
SHARD = "model.safetensors-00001-of-00001.safetensors"
REC_SHA = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
DON_SHA = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
REC_REV = "2fc06364715b967f1860aea9cf38778875588b17"
DON_REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATA = {"carry": ("synthetic_dev.jsonl", "c2e41927daabc5ad734fe67fa1e2262ef2b7631e00226bcfea1815d42d7bbd5e"),
        "gsm8k": ("gsm8k_dev.jsonl", "6e2fbfa175bf52e49f96318d7ceea642e1e13aa9b9e80d8e4fe02d998614b13f")}


def data_path(family):
    return ROOT / "data/development" / DATA[family][0]


def items(start, stop):
    for family, (_, expected) in DATA.items():
        path = data_path(family)
        if sha256_file(path) != expected:
            raise ValueError(f"Development checksum mismatch for {path}")
        rows = load_dataset_rows(path)
        if family == "carry":
            rows = [r for r in rows if r["subskill"] == "carry_add_sub"]
        assert len(rows) >= stop and all(r["split"] == "dev" for r in rows)
        for r in rows[start:stop]:
            yield family, r


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def schema(model):
    def tensors(named):
        return [(n, list(t.shape), str(t.dtype)) for n, t in named]
    graph = {"modules": [(n, type(m).__module__ + "." + type(m).__qualname__)
                         for n, m in model.named_modules()],
             "parameters": tensors(model.named_parameters(remove_duplicate=False)),
             "buffers": tensors(model.named_buffers(remove_duplicate=False))}
    param_aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        param_aliases.setdefault(id(parameter), []).append(name)
    graph["aliases"] = sorted(names for names in param_aliases.values() if len(names) > 1)
    return {"graph_sha256": hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
            "modules": len(graph["modules"]), "text_layers": model.config.text_config.num_hidden_layers,
            "runtime_parameters": sum(p.numel() for p in model.parameters()),
            "tied_embeddings": model.get_input_embeddings().weight is model.get_output_embeddings().weight}

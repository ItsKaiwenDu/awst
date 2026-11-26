"""Evaluate the fixed H002 screen or its disjoint remainder on the local recipient."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_interface.tools import AgentWeightToolbox
from src.evaluation.metrics import detect_repetition_loop
from src.evaluation.parser import answers_match
from src.evaluation.runner import format_prompt, load_dataset_rows, load_model_and_tokenizer

RECIPIENT = ROOT / "hf_cache/models/qwen3.5_0.8b"
EDITED = ROOT / "hf_cache/edited"
RUN = Path(__file__).resolve().parent
DATASETS = {
    "carry_add_sub": ROOT / "data/development/synthetic_dev.jsonl",
    "gsm8k_dev": ROOT / "data/development/gsm8k_dev.jsonl",
}
LABELS = ("c0", "c1_low", "c1_high", "t1_low", "t1_high")


def evaluate(label: str, phase: str) -> None:
    checkpoint = RECIPIENT if label == "c0" else EDITED / f"h002_{label}"
    summary_path = RUN / f"{label}_{phase}.json"
    log_path = ROOT / f"runs/logs/h002_{label}_{phase}.jsonl"
    if summary_path.exists() or log_path.exists():
        raise FileExistsError(f"Evidence already exists for {label}/{phase}")
    started = time.perf_counter()
    model, tokenizer, device, load_sec = load_model_and_tokenizer(checkpoint, "mps", "bfloat16")
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
    assert model.config.text_config.num_hidden_layers == 24
    # The frozen Transformers class omits the 15 serialized multi-token prediction tensors.
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == 852_985_920
    schema = [(n, list(p.shape), str(p.dtype)) for n, p in model.named_parameters()]
    schema_hash = hashlib.sha256(json.dumps(schema).encode()).hexdigest()
    # This process loads only a recipient. Even T1 finalist scoring has no donor handle.
    meter = AgentWeightToolbox(donor_dir=None, recipient_dir=checkpoint, track="C1")
    if label != "c0":
        meter.record_dev_trial()
    nforward = 0

    def count_forward(module, args):
        nonlocal nforward
        nforward += 1

    handle = model.register_forward_pre_hook(count_forward)
    results = {}
    with log_path.open("x", encoding="utf-8", buffering=1) as log, torch.inference_mode():
        for name, path in DATASETS.items():
            items = load_dataset_rows(path)
            if name == "carry_add_sub":
                items = [r for r in items if r.get("subskill") == name]
            assert len(items) == (300 if name == "carry_add_sub" else 500)
            items = items[:20] if phase == "screen" else items[20:]
            correct = truncated = loops = unparsed = tokens = 0
            at_start = nforward
            for item in items:
                inputs = tokenizer(format_prompt(tokenizer, item["prompt"], False),
                                   return_tensors="pt").to(device)
                prompt_len = inputs["input_ids"].shape[1]
                before = nforward
                output = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
                ids = output[0, prompt_len:].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True)
                ok, prediction, method = answers_match(text, str(item["answer"]))
                cutoff = len(ids) == 1024 and ids[-1] != tokenizer.eos_token_id
                loop = detect_repetition_loop(text)
                used = nforward - before
                if not used:
                    raise RuntimeError("Recipient forward hook counted no calls")
                meter.record_forward_pass("recipient", path, count=used)
                correct += int(ok)
                truncated += int(cutoff)
                loops += int(loop)
                unparsed += int(prediction is None)
                tokens += len(ids)
                log.write(json.dumps({"id": item["id"], "dataset": name, "correct": ok,
                                      "prediction": prediction, "method": method, "truncated": cutoff,
                                      "loop": loop, "tokens": len(ids), "forward_calls": used,
                                      "response": text}, ensure_ascii=False) + "\n")
            results[name] = {"n": len(items), "correct": correct, "truncated": truncated,
                             "loop": loops, "unparsed": unparsed, "generated_tokens": tokens,
                             "recipient_forward_passes": nforward - at_start}
            print(label, phase, name, correct, "/", len(items), flush=True)
    handle.remove()
    assert nforward == meter.recipient_forward_passes
    summary = {"label": label, "phase": phase, "datasets": results, "device": device,
               "dtype": "bfloat16", "enable_thinking": False, "do_sample": False,
               "max_new_tokens": 1024, "model_class": type(model).__name__,
               "tied_embeddings": True, "runtime_parameters": parameter_count,
               "runtime_schema_sha256": schema_hash, "recipient_forward_passes": nforward,
               "donor_forward_passes": 0, "dev_trials": meter.dev_trials_used,
               "load_seconds": load_sec, "wall_seconds": time.perf_counter() - started}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=LABELS)
    parser.add_argument("phase", choices=("screen", "remainder"))
    args = parser.parse_args()
    evaluate(args.label, args.phase)

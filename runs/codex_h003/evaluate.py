"""Score the fixed T2 screen and disjoint follow-up; resume only missing examples."""

import fcntl
import gc
import json
import os
import time
from datetime import datetime, timezone

import torch

from common import (DONOR, EDITED, RECIPIENT, REC_SHA, ROOT, RUN, SHARD, data_path, items,
                    schema, sha256_file, write_json)
from src.agent_interface.tools import AgentWeightToolbox
from src.evaluation.metrics import compute_eval_metrics, detect_repetition_loop
from src.evaluation.parser import answers_match
from src.evaluation.runner import format_prompt, load_model_and_tokenizer

RANGES = {"screen": (16, 36), "followup": (36, 100)}
LOGS = ROOT / "runs/logs"
STRENGTHS = {"half": 0.5, "full": 1.0}
TRIALS = RUN / "trial_starts.jsonl"


def read_lines(path):
    if path.exists():
        with path.open() as stream:
            return [json.loads(line) for line in stream]
    return []


def score(label, phase):
    manifest = RUN / f"{label}_{phase}.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    checkpoint = {"c0": RECIPIENT, "c4": EDITED / "m2_c4_act"}.get(label, EDITED / f"h003_{label}")
    digest = sha256_file(checkpoint / SHARD)
    if label == "c0":
        assert digest == REC_SHA
    elif label in STRENGTHS:
        assert digest == json.loads((RUN / f"{label}_patch.json").read_text())["edited_shard_sha256"]
    else:
        assert label == "c4"
    log = LOGS / f"h003_{label}_{phase}.jsonl"
    records = read_lines(log)
    rows = list(items(*RANGES[phase]))
    assert [(r["family"], r["id"]) for r in records] == [(f, r["id"]) for f, r in rows[:len(records)]]
    assert all(r["checkpoint_sha256"] == digest for r in records)
    previous = read_lines(TRIALS)
    meter = AgentWeightToolbox(DONOR, checkpoint, track="T2", max_dev_trials=12)
    for event in previous:
        if event["candidate_trial"]:
            meter.record_dev_trial()
    if label in STRENGTHS:
        meter.record_dev_trial()
    started_at = datetime.now(timezone.utc).isoformat()
    with TRIALS.open("a") as f:
        f.write(json.dumps({"utc": started_at, "label": label, "phase": phase, "candidate_trial": label in STRENGTHS,
                            "resume_completed_rows": len(records), "checkpoint_sha256": digest}) + "\n")
    started = time.perf_counter()
    model, tokenizer, device, load_seconds = load_model_and_tokenizer(checkpoint, "mps", "bfloat16")
    runtime = schema(model)
    assert runtime["runtime_parameters"] == 852985920 and runtime["tied_embeddings"]
    counter = 0
    def count(_module, _args):
        nonlocal counter
        counter += 1
    prior = len(records)
    with model.register_forward_pre_hook(count), torch.inference_mode(), log.open("a", buffering=1) as f:
        for family, item in rows[prior:]:
            t0 = time.perf_counter()
            enc = tokenizer(format_prompt(tokenizer, item["prompt"], False), return_tensors="pt").to(device)
            size = int(enc.input_ids.shape[1])
            before = counter
            out = model.generate(**enc, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            ids = out[0, size:].tolist()
            response = tokenizer.decode(ids, skip_special_tokens=True)
            ok, pred, method = answers_match(response, str(item["answer"]))
            calls = counter - before
            assert calls > 0
            meter.record_forward_pass("recipient", data_path(family), calls)
            row = {"family": family, "id": item["id"], "correct": bool(ok),
                   "truncated": len(ids) == 1024 and ids[-1] != tokenizer.eos_token_id,
                   "repetition_loop": bool(detect_repetition_loop(response)), "parse_valid": pred is not None,
                   "input_tokens": size, "generated_tokens": len(ids), "recipient_forward_passes": calls,
                   "latency_sec": time.perf_counter() - t0, "prediction": pred, "parse_method": method,
                   "response": response, "checkpoint_sha256": digest}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.append(row)
    assert counter == meter.recipient_forward_passes
    counts = {}
    for family in ("carry", "gsm8k"):
        scored = [r for r in records if r["family"] == family]
        counts[family] = compute_eval_metrics(scored)
        counts[family]["correct_ids"] = [r["id"] for r in scored if r["correct"]]
        counts[family]["ids"] = [r["id"] for r in scored]
    result = {"track": "T2", "label": label, "phase": phase, "checkpoint_sha256": digest, "datasets": counts,
              "device": device, "dtype": "bfloat16", "do_sample": False, "enable_thinking": False,
              "max_new_tokens": 1024, "runtime": runtime, "started_utc": started_at,
              "load_seconds": load_seconds, "wall_seconds": time.perf_counter()-started,
              "recipient_forward_passes": sum(r["recipient_forward_passes"] for r in records), "donor_forward_passes": 0,
              "input_tokens": sum(r["input_tokens"] for r in records),
              "generated_tokens": sum(r["generated_tokens"] for r in records),
              "prompt_generations": len(records), "mps_driver_allocated_bytes_at_end": torch.mps.driver_allocated_memory(),
              "local_log": str(log.relative_to(ROOT)), "log_sha256": sha256_file(log)}
    write_json(manifest, result)
    print(label, phase, [v["correct_count"] for v in counts.values()], flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return result


def run():
    screens = {label: score(label, "screen") for label in ("c0", "c4", "half", "full")}
    def key(label):
        scores = screens[label]["datasets"]
        return (sum(row["correct_count"] for row in scores.values()), scores["carry"]["correct_count"], -STRENGTHS[label])
    winner = max(STRENGTHS, key=key)
    selection = {"track": "T2", "label": winner, "strength": STRENGTHS[winner],
                 "localization_sha256": sha256_file(RUN / "localization.json"),
                 "checkpoint_sha256": screens[winner]["checkpoint_sha256"],
                 "selection_keys": {label: key(label) for label in STRENGTHS},
                 "rule": "total correct, then carry correct, then smaller strength"}
    path = RUN / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(selection))
    else:
        write_json(path, selection)
    print("Frozen finalist", winner, flush=True)
    for label in ("c0", "c4", winner):
        score(label, "followup")


if __name__ == "__main__":
    with (LOGS / "h003_evaluation_lock.tmp").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

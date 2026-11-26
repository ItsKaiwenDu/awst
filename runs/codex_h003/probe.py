"""Measure fixed T2 head fingerprints, then ablate the two shortlisted heads."""

import fcntl
import gc
import hashlib
import json
import os
import re
import time
from contextlib import ExitStack
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from common import (DONOR, DON_SHA, RECIPIENT, REC_SHA, ROOT, RUN, SHARD,
                    data_path, items, sha256_file, write_json)
from src.agent_interface.tools import AgentWeightToolbox
from src.evaluation.runner import format_prompt, load_model_and_tokenizer

LAYERS = [i for i in range(8, 23) if i % 4 != 3]
CACHE = ROOT / "hf_cache/h003_probe_cache"
LEDGER = RUN / "probe_calls.jsonl"
CUE = r"The final answer is \boxed{"


def encode(tokenizer, item):
    text = format_prompt(tokenizer, item["prompt"], False) + CUE
    start = text.index(item["prompt"], text.index("Problem: "))
    stop = start + len(item["prompt"])
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    pos, digits = [], []
    for i, (left, right) in enumerate(offsets):
        token = text[max(start, left):min(right, stop)] if left < stop and right > start else ""
        if token.strip():
            pos.append(i)
            digits.append(any(c in "0123456789" for c in token))
    if not any(digits) or all(digits):
        raise ValueError(f"No digit/non-digit token contrast for {item['id']}")
    return enc, pos, np.array(digits)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def measure(model, tokenizer, role, layer_head, family, item, meter):
    variant = "clean" if layer_head is None else "L%d_H%d" % layer_head
    path = CACHE / f"{role}_{variant}_{item['id']}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            return dict(cache)
    enc, positions, digits = encode(tokenizer, item)
    digit_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)]
    assert all(len(ids) == 1 for ids in digit_ids)
    gold = int(re.search(r"[0-9]", str(item["answer"]))[0])
    gates = {}
    layers = model.model.language_model.layers
    def capture(key):
        def hook(_module, _args, output):
            gates[key] = output[0, positions].float().cpu().numpy()
        return hook

    def magnitude(layer):
        def hook(_module, args):
            routed = args[0][0, positions].float().reshape(len(positions), 16, 128)
            gates[layer, "rms"] = routed.square().mean(-1).sqrt().cpu().numpy()
        return hook

    def zero_head(_module, args):
        h = layer_head[1]
        routed = args[0].clone()
        routed[..., 128*h:128*(h+1)] = 0
        return (routed, *args[1:])

    def charge(_module, _args):
        if role == "donor" and meter.donor_forward_passes >= 64:
            raise RuntimeError("H003 donor forward limit reached")
        meter.record_forward_pass(role, data_path(family))
        event = {"role": role, "variant": variant, "dataset": str(data_path(family).relative_to(ROOT)),
                 "id": item["id"], "input_tokens": int(enc.input_ids.shape[1]),
                 "utc": datetime.now(timezone.utc).isoformat()}
        with LEDGER.open("a") as log:
            log.write(json.dumps(event) + "\n")
            log.flush()
            os.fsync(log.fileno())

    started = time.perf_counter()
    with ExitStack() as handles, torch.inference_mode():
        handles.enter_context(model.register_forward_pre_hook(charge))
        if layer_head is None:
            for l in LAYERS:
                lin = layers[l].linear_attn
                handles.enter_context(lin.in_proj_a.register_forward_hook(capture((l, "a"))))
                handles.enter_context(lin.in_proj_b.register_forward_hook(capture((l, "b"))))
                handles.enter_context(lin.out_proj.register_forward_pre_hook(magnitude(l)))
        else:
            handles.enter_context(layers[layer_head[0]].linear_attn.out_proj.register_forward_pre_hook(zero_head))
        digest = tensor_digest(enc.input_ids)
        logits = model(**enc.to("mps"), use_cache=False, logits_to_keep=1).logits
        scores = logits[0, -1, [ids[0] for ids in digit_ids]].float().cpu().numpy()
    result = {"margin": scores[gold] - np.delete(scores, gold).max(),
              "ids_sha256": digest, "digits": digits, "secs": time.perf_counter() - started}
    if layer_head is None:
        feature = []
        for l in LAYERS:
            lin = layers[l].linear_attn
            a, b = gates[l, "a"], gates[l, "b"]
            decay = lin.A_log.detach().float().cpu().numpy() + np.log(
                np.maximum(np.logaddexp(0, a + lin.dt_bias.detach().float().cpu().numpy()), 1e-30))
            beta = torch.from_numpy(b).sigmoid().numpy()
            feature.append(np.stack((decay, beta, gates[l, "rms"], a), axis=-1))
        result["features"] = np.stack(feature)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **result)
    os.replace(tmp, path)
    return result


def correlations(x, y):
    def columns(a):
        a = a.astype(np.float64)
        centered = a - a.mean(axis=0)
        return centered / np.maximum(np.linalg.norm(centered, axis=0), 1e-12)
    return columns(x).T @ columns(y)


def short_list(clean, rows):
    groups = np.array([f for f, r in rows])
    advantage = np.array([d["margin"] > r["margin"] for d, r in zip(clean["donor"], clean["recipient"])])
    ranks = []
    for i, layer in enumerate(LAYERS):
        ds, rs = ([x["features"][i] for x in clean[role]] for role in ("donor", "recipient"))
        d, r = np.concatenate(ds), np.concatenate(rs)
        corr = (correlations(d[:, :, 0], r[:, :, 0]) + correlations(d[:, :, 1], r[:, :, 1])) / 2
        dhs, rhs = linear_sum_assignment(-corr)
        def selectivity(xs):
            return np.stack([np.log((v[m, :, 2].mean(0) + 1e-12) / (v[~m, :, 2].mean(0) + 1e-12))
                             for v, m in zip(xs, (x["digits"] for x in clean["donor"]))])
        difference = selectivity(ds)[:, dhs] - selectivity(rs)[:, rhs]
        means = []
        for group in ("carry", "gsm8k"):
            mask = (groups == group) & advantage
            means.append(difference[mask if mask.any() else groups == group].mean(0))
        for j, (dh, rh) in enumerate(zip(dhs, rhs)):
            corr_value = float(corr[dh, rh])
            ranks.append({"layer": layer, "donor_head": int(dh), "recipient_head": int(rh),
                          "correlation": corr_value, "selectivity_gap_carry": float(means[0][j]),
                          "selectivity_gap_gsm8k": float(means[1][j]),
                          "rank_score": float((means[0][j] + means[1][j]) / 2 * max(0, corr_value))})
    ordered = sorted(ranks, key=lambda q: (-q["rank_score"], q["layer"], q["recipient_head"]))
    best = [ordered[0], next(q for q in ordered if q["layer"] != ordered[0]["layer"])]
    return best, ordered, {g: int(advantage[groups == g].sum()) for g in ("carry", "gsm8k")}


def collect():
    rows = list(items(0, 16))
    clean, shortlist, rankings = {}, [], []
    meter = AgentWeightToolbox(DONOR, RECIPIENT, track="T2")
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            e = json.loads(line)
            meter.record_forward_pass(e["role"], ROOT / e["dataset"])
    started = time.perf_counter()
    probe_indices = [i for i in range(len(rows)) if i % 16 < 6]
    for phase in ("clean", "ablate"):
        for role, model_dir, expected in (("recipient", RECIPIENT, REC_SHA), ("donor", DONOR, DON_SHA)):
            if sha256_file(model_dir / SHARD) != expected:
                raise ValueError(f"Unexpected {role} checkpoint")
            model, tokenizer, _, _ = load_model_and_tokenizer(model_dir, "mps", "bfloat16")
            if phase == "clean":
                clean[role] = [measure(model, tokenizer, role, None, f, r, meter) for f, r in rows]
            else:
                for selection in shortlist:
                    key = (selection["layer"], selection[role + "_head"])
                    drops = [float(clean[role][i]["margin"] -
                                   measure(model, tokenizer, role, key, *rows[i], meter)["margin"])
                             for i in probe_indices]
                    selection[role + "_margin_drops"] = drops
            del model, tokenizer
            gc.collect()
            torch.mps.empty_cache()
        if phase == "clean":
            for d, r in zip(clean["donor"], clean["recipient"]):
                assert str(d["ids_sha256"]) == str(r["ids_sha256"]) and np.array_equal(d["digits"], r["digits"])
            shortlist, rankings, advantage = short_list(clean, rows)
    for selection in shortlist:
        selection["causal_score"] = float(np.mean(selection["donor_margin_drops"]) -
                                           np.mean(selection["recipient_margin_drops"]))
    selected = sorted(shortlist, key=lambda s: (-s["causal_score"], s["layer"], s["recipient_head"]))[0].copy()
    for role in ("donor", "recipient"):
        means = [v["features"][LAYERS.index(selected["layer"]), v["digits"], selected[role + "_head"], 3].mean()
                 for v in clean[role]]
        selected[role + "_digit_a_median"] = float(np.median(means))
    margins = [{"id": row["id"], "family": fam, **{role: float(clean[role][i]["margin"]) for role in clean}}
               for i, (fam, row) in enumerate(rows)]
    write_json(RUN / "localization.json", {"track": "T2", "datapaths": [str(data_path(f).relative_to(ROOT))
               for f in ("carry","gsm8k")], "candidate_layers": LAYERS, "digit_margin_advantage_n": advantage,
               "clean_margins": margins, "ablation_ids": [rows[i][1]["id"] for i in probe_indices],
               "shortlist": shortlist, "selected": selected, "ranked_pairs": rankings,
               "meter": meter.save_budget_report(), "wall_seconds": time.perf_counter() - started,
               "donor_shard_sha256": DON_SHA, "recipient_shard_sha256": REC_SHA})
    print("Selected", selected["layer"], selected["donor_head"], selected["recipient_head"],
          "forwards", meter.donor_forward_passes, meter.recipient_forward_passes)


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / "localization.json").exists():
            raise FileExistsError("H003 localization already frozen")
        collect()

"""Reconcile the seven score manifests and compute paired follow-up effects."""

import json
from collections import Counter

import numpy as np
from scipy.stats import binomtest

from common import DONOR, ROOT, RUN, SHARD, sha256_file, write_json


def load(name):
    return json.loads((RUN / f"{name}.json").read_text())


def paired(treatment, control, rng):
    assert treatment["ids"] == control["ids"] and len(set(treatment["ids"])) == len(treatment["ids"])
    t, c = map(set, (treatment["correct_ids"], control["correct_ids"]))
    delta = np.array([int(i in t) - int(i in c) for i in treatment["ids"]])
    n = len(delta)
    fixed, lost = len(t-c), len(c-t)
    means = np.mean(rng.choice(delta, size=(10000, n), replace=True), axis=1)
    return {"n": n, "fixed": fixed, "lost": lost, "both_correct": len(t & c), "both_wrong": n-len(t | c),
            "delta_pp": 100*float(delta.mean()), "paired_bootstrap95_pp": (100*np.quantile(means, [0.025,0.975])).tolist(),
            "mcnemar_exact_p": binomtest(fixed, fixed+lost, 0.5).pvalue if fixed+lost else 1.0}


def summarize():
    selection = load("selection")
    winner = selection["label"]
    refs = ["c0","c4"]
    groups = {"screen": [*refs, "half", "full"], "followup": [*refs, winner]}
    rng = np.random.default_rng(42)
    scores, contrasts, manifests = {}, {}, []
    for phase, arms in groups.items():
        scores[phase], contrasts[phase], phase_rows = {}, {}, {}
        for arm in arms:
            m = load(f"{arm}_{phase}")
            raw_log = ROOT / m["local_log"]
            assert sha256_file(raw_log) == m["log_sha256"]
            if arm == "c4":
                assert m["checkpoint_sha256"] == load("c4_reference")["shard_sha256"]
            if arm == winner:
                assert m["checkpoint_sha256"] == selection["checkpoint_sha256"]
            with raw_log.open() as f:
                logs = [json.loads(line) for line in f]
            for field in ("input_tokens","generated_tokens","recipient_forward_passes"):
                assert sum(r[field] for r in logs) == m[field]
            m_signatures = {r["id"]: r for r in logs}
            scores[phase][arm] = {}
            for family, stats in m["datasets"].items():
                assert set(stats["correct_ids"]) == {i for i in stats["ids"] if m_signatures[i]["correct"]}
                scores[phase][arm][family] = {k: stats[k] for k in ("n","correct_count","pass_at_1","truncation_rate",
                                                                    "repetition_loop_rate","invalid_parse_rate")}
            phase_rows[arm] = m["datasets"]
            manifests.append(m)
        for arm in arms:
            if arm in refs:
                continue
            for ref in refs:
                contrasts[phase][arm+"_vs_"+ref] = {family: paired(stats, phase_rows[ref][family], rng)
                                                    for family, stats in phase_rows[arm].items()}

    calls = [json.loads(line) for line in (RUN / "probe_calls.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (RUN / "trial_starts.jsonl").read_text().splitlines()]
    unique_calls = {(c["role"],c["variant"],c["id"]) for c in calls}
    completed = {(m["label"], m["phase"]) for m in manifests}
    assert len(completed) == 7 and completed == {(e["label"], e["phase"]) for e in events}
    trials = sum(e["candidate_trial"] for e in events)
    assert trials <= 12
    assert len({m["runtime"]["graph_sha256"] for m in manifests}) == 1
    calibration = {c["id"] for c in calls}
    screened = {i for stats in load("c0_screen")["datasets"].values() for i in stats["ids"]}
    followed = {i for stats in load("c0_followup")["datasets"].values() for i in stats["ids"]}
    assert calibration.isdisjoint(screened | followed) and screened.isdisjoint(followed)
    probe = load("localization")
    for role in ("donor","recipient"):
        assert sum(c["role"] == role for c in calls) == probe["meter"][role+"_forward_passes"]
    assert probe["meter"]["donor_forward_passes"] <= 64
    result = {"track": "T2", "selection": winner, "scores": scores, "contrasts": contrasts,
              "paired_ci": "10,000 paired problem bootstrap draws, NumPy default_rng(42), loops in script order; descriptive",
              "probe_by_role": {role: {"forwards": sum(c["role"] == role for c in calls),
                    "input_tokens": sum(c["input_tokens"] for c in calls if c["role"] == role),
                    "variants": dict(Counter(c["variant"] for c in calls if c["role"] == role))}
                    for role in ("recipient", "donor")}, "calibration_seconds": probe["wall_seconds"],
              "candidate_trials_used": trials, "reference_evaluations": len(events)-trials,
              "repeated_probe_calls": len(calls)-len(unique_calls), "restarted_evaluations": len(events)-len(completed),
              "generations": sum(m["prompt_generations"] for m in manifests),
              "generated_tokens": sum(m["generated_tokens"] for m in manifests),
              "generation_prompt_tokens": sum(m["input_tokens"] for m in manifests),
              "generation_recipient_forward_passes": sum(m["recipient_forward_passes"] for m in manifests),
              "donor_forward_passes": probe["meter"]["donor_forward_passes"],
              "recipient_forward_passes": probe["meter"]["recipient_forward_passes"] + sum(m["recipient_forward_passes"] for m in manifests),
              "evaluation_seconds": sum(m["wall_seconds"] for m in manifests),
              "build_seconds": sum(load(label+"_patch")["wall_seconds"] for label in ("half","full")),
              "gradient_steps": 0, "donor_local_shard_bytes": (DONOR / SHARD).stat().st_size,
              "donor_model_loads_for_probing": 2, "runtime": manifests[0]["runtime"]}
    write_json(RUN / "dev_results.json", result)


if __name__ == "__main__":
    summarize()

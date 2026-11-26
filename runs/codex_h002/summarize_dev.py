"""Check disjoint H002 panels and make portable paired development results."""

import hashlib
import json
from pathlib import Path

from runs.codex_h002.evaluate_dev import DATASETS, ROOT, RUN
from src.evaluation.runner import load_dataset_rows


def summary(label, phase):
    return json.loads((RUN / f"{label}_{phase}.json").read_text())


def winner(arm):
    def count(strength):
        result = summary(f"{arm}_{strength}", "screen")
        return sum(v["correct"] for v in result["datasets"].values())

    return f"{arm}_low" if count("low") >= count("high") else f"{arm}_high"


def load_predictions(label, phase, name, expected_ids):
    path = ROOT / f"runs/logs/h002_{label}_{phase}.jsonl"
    with path.open(encoding="utf-8") as stream:
        rows = [r for r in map(json.loads, stream) if r["dataset"] == name]
    if [r["id"] for r in rows] != expected_ids:
        raise AssertionError(f"Misordered or missing examples for {label}/{phase}/{name}")
    data = summary(label, phase)["datasets"][name]
    if (sum(r["correct"] for r in rows) != data["correct"]
            or sum(r["tokens"] for r in rows) != data["generated_tokens"]
            or sum(r["forward_calls"] for r in rows) != data["recipient_forward_passes"]):
        raise AssertionError(f"Manifest and row log disagree for {label}/{phase}/{name}")
    return [r["correct"] for r in rows]


def paired(reference, comparison):
    pairs = list(zip(reference, comparison, strict=True))
    return {"improved": sum(not a and b for a, b in pairs),
            "regressed": sum(a and not b for a, b in pairs),
            "both_correct": sum(a and b for a, b in pairs),
            "both_incorrect": sum(not a and not b for a, b in pairs)}


def summarize():
    finalists = ("c0", winner("c1"), winner("t1"))
    labels = ("c0", "c1_low", "c1_high", "t1_low", "t1_high")
    results = {"finalists": finalists, "datasets": {}}
    for name, path in DATASETS.items():
        ids = [r["id"] for r in load_dataset_rows(path)
               if name == "gsm8k_dev" or r.get("subskill") == "carry_add_sub"]
        if len(set(ids)) != len(ids):
            raise AssertionError("Duplicate development IDs")
        screen = {l: load_predictions(l, "screen", name, ids[:20]) for l in labels}
        remainder = {l: load_predictions(l, "remainder", name, ids[20:]) for l in finalists}
        full = {l: screen[l] + remainder[l] for l in finalists}
        data = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for panel, records in [("screen", screen), ("remainder", remainder), ("full", full)]:
            data[panel] = {
                "n": len(records["c0"]), "correct": {k: sum(v) for k, v in records.items()},
                "paired": {f"{a}_to_{b}": paired(records[a], records[b])
                           for a, b in [(finalists[0], finalists[1]),
                                        (finalists[0], finalists[2]),
                                        (finalists[1], finalists[2])]},
            }
            if panel != "full":
                # One bit per problem in source-file order. No responses or answer text.
                data[panel]["correct_bits"] = {
                    l: "".join("1" if ok else "0" for ok in values)
                    for l, values in records.items()
                }
        results["datasets"][name] = data
    return results


if __name__ == "__main__":
    target = Path(RUN) / "dev_results.json"
    target.write_text(json.dumps(summarize(), indent=2) + "\n", encoding="utf-8")

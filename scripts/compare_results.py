from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_numbers(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from flatten_numbers(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) for x in obj):
            for idx, value in enumerate(obj):
                yield f"{prefix}[{idx}]", float(value)
        else:
            for idx, value in enumerate(obj):
                yield from flatten_numbers(value, f"{prefix}[{idx}]")
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    reference = dict(flatten_numbers(load_json(args.reference)))
    candidate = dict(flatten_numbers(load_json(args.candidate)))
    shared = sorted(set(reference) & set(candidate))
    diffs = []
    for key in shared:
        ref = reference[key]
        cand = candidate[key]
        abs_diff = abs(ref - cand)
        rel_diff = abs_diff / (abs(ref) + 1e-12)
        diffs.append((abs_diff, rel_diff, key, ref, cand))
    diffs.sort(reverse=True)
    print(json.dumps([
        {
            "key": key,
            "reference": ref,
            "candidate": cand,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
        }
        for abs_diff, rel_diff, key, ref, cand in diffs[: args.top_k]
    ], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

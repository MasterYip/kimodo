#!/usr/bin/env python3
"""Fingerprint historical Kimodo output datasets without mutating them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(dataset: Path) -> dict:
    manifest = dataset / "manifest.csv"
    row = {
        "dataset_name": dataset.name,
        "dataset_path": str(dataset.resolve()),
        "status": "manifest_missing" if not manifest.is_file() else "ok",
    }
    if not manifest.is_file():
        return row
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths = [dataset / item.get("path", "") for item in rows]
    existing = [path for path in paths if path.is_file()]
    fps = Counter()
    schema = Counter()
    total_bytes = 0
    for path in existing:
        total_bytes += path.stat().st_size
        try:
            with np.load(path, allow_pickle=False) as archive:
                fps[str(float(np.asarray(archive["fps"]).reshape(-1)[0]))] += 1
                schema["|".join(sorted(archive.files))] += 1
        except Exception as exc:
            schema[f"ERROR:{type(exc).__name__}"] += 1
    row.update({
        "manifest_sha256": sha256(manifest),
        "manifest_rows": len(rows),
        "npz_paths_existing": len(existing),
        "npz_paths_missing": len(paths) - len(existing),
        "npz_bytes": total_bytes,
        "motion_type_counts": json.dumps(Counter(item.get("motion_type", "") for item in rows), sort_keys=True),
        "fps_counts": json.dumps(fps, sort_keys=True),
        "schema_counts": json.dumps(schema, sort_keys=True),
    })
    if len(existing) != len(paths):
        row["status"] = "incomplete"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    datasets = sorted({path.parent for path in root.glob("*/manifest.csv")})
    rows = [audit(dataset) for dataset in datasets]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    fields = sorted({key for row in rows for key in row})
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"datasets": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
                      "incomplete": sum(r["status"] != "ok" for r in rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

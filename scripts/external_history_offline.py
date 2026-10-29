# SPDX-FileCopyrightText: Copyright (c) 2026 DataDesigner contributors
# SPDX-License-Identifier: Apache-2.0

"""Low-level offline recipe entry, driven by the upstream discovery/seed manifest.

Consumes every row of a supplied normalized seed file; it is NOT a raw-log
reader or a replacement discovery pass. Does not generate/repair model answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import data_designer.config as dd
from data_designer.integrations.external_history import offline_config
from data_designer.interface import DataDesigner


def run_offline(seed: Path, artifact_root: Path, dataset_name: str) -> dict[str, Any]:
    """Use the real NeMo executor and preserve all supplied rows, including failures."""
    raw = seed.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("seed_contains_zero_records")
    providers = Counter(json.loads(row["packet_json"])["provider"] for row in rows)
    result = DataDesigner(
        artifact_path=artifact_root,
        model_providers=[dd.ModelProvider(name="offline-unused", endpoint="http://127.0.0.1:9/v1")],
    ).create(offline_config(seed), num_records=len(rows), dataset_name=dataset_name)
    frame = result.load_dataset()
    if len(frame) != len(rows) or sorted(frame["packet_json"]) != sorted(row["packet_json"] for row in rows):
        raise ValueError("seed_occurrence_round_trip_mismatch")
    counts = Counter(frame["binding_disposition"])
    undecidable = []
    for row in frame.to_dict("records"):
        if row["binding_disposition"] == "undecidable":
            validation = row["binding_validation"]
            undecidable.append(
                {
                    "packet_id": json.loads(row["packet_json"])["packet_id"],
                    "reasons": list(validation["reasons"]),
                    "evidence_to_settle": list(validation["evidence_needed"]),
                }
            )
    return {
        "UTC": datetime.now(UTC).isoformat(),
        "scope": "offline binding validation of supplied extractions,NOT semantic generation or training",
        "seed_sha256": hashlib.sha256(raw).hexdigest(),
        "native_artifact_path": str(result.artifact_storage.base_dataset_path),
        "packet_rows": len(rows),
        "record_occurrences_in_packets_including_repeats": sum(
            len(json.loads(row["packet_json"]).get("occurrences", [])) for row in rows
        ),
        "providers": dict(providers),
        "mechanical_stage": {key: counts[key] for key in ["admitted", "excluded", "undecidable"]},
        "undecidable": undecidable,
        "content_review": "NOT_RUN;mechanically admitted rows await original-source review",
        "training_admitted": 0,
        "new_model_calls": 0,
        "raw_file_authenticity_check": "NOT_PERFORMED;bindings are to supplied native objects",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-jsonl", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="external_history_offline")
    args = parser.parse_args()
    report = run_offline(args.seed_jsonl, args.artifact_root, args.dataset_name)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    target = args.artifact_root / f"offline-report-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(target), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

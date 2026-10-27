# SPDX-FileCopyrightText: Copyright (c) 2026 DataDesigner contributors
# SPDX-License-Identifier: Apache-2.0

"""Drive all existing discovery occurrences into native NeMo metadata seeds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import data_designer.config as dd
from data_designer.integrations.external_history_seeds import DerivedInputs, export_seeds, seed_manifest_config
from data_designer.interface import DataDesigner


def native_roundtrip(seed: Path, artifacts: Path, expected_rows: int) -> dict[str, Any]:
    designer = DataDesigner(
        artifact_path=artifacts,
        model_providers=[dd.ModelProvider(name="offline-unused", endpoint="http://127.0.0.1:9/v1")],
    )
    designer.set_run_config(dd.RunConfig(buffer_size=10000, otel_metrics_port=None, display_tui=False))
    result = designer.create(seed_manifest_config(seed), num_records=expected_rows, dataset_name="reader_seed_manifest")
    frame = result.load_dataset()
    if len(frame) != expected_rows:
        raise ValueError("native_seed_row_count_mismatch")
    # Compare every ordered seed string, not just row counts or a human-selected sample.
    with seed.open() as stream:
        for expected, actual in zip(stream, frame["seed_json"], strict=True):
            if json.loads(expected)["seed_json"] != actual:
                raise ValueError("native_seed_identity_order_or_bytes_changed")
    counts = Counter(item["disposition"] for item in frame["seed_validation"])
    return {
        "native_artifact_path": str(result.artifact_storage.base_dataset_path),
        "seed_rows": len(frame),
        "metadata_validation_counts": dict(counts),
        "payload_or_semantic_validation": False,
        "training_admitted": 0,
        "model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-roundtrip", action="store_true")
    args = parser.parse_args()
    report = export_seeds(DerivedInputs.from_preparation(args.preparation_root), args.output_dir)
    if args.native_roundtrip:
        native = native_roundtrip(
            args.output_dir / "seeds.jsonl", args.output_dir / "native-artifacts", report["seed_rows"]
        )
        (args.output_dir / "native-roundtrip.json").write_text(json.dumps(native, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["needs", "input_hashes"]}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()

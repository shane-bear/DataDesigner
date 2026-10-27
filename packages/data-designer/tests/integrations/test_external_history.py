# SPDX-FileCopyrightText: Copyright (c) 2026 DataDesigner contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

import data_designer.config as dd
import data_designer.lazy_heavy_imports as lazy
from data_designer.integrations.external_history import (
    EvidencePacket,
    Occurrence,
    canonical_json,
    generation_config,
    json_pointer,
    native_object_sha256,
    offline_config,
    packet_from_occurrences,
    validate_bindings,
    validate_generated_dataframe,
    value_type,
)
from data_designer.interface import DataDesigner


def seed_pair(
    value: Any = 'echo "quoted"\\path\nnext', provider: str = "claude_code"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synthetic fixture only; neither real history nor a reference model answer."""
    native = {"payload": {"value": value}}
    packet = {
        "packet_id": f"DEMO-{provider}",
        "provider": provider,
        "source_group": f"arbitrary-source/{provider}",
        "sensitivity": "safe",
        "cutoff_order": 2,
        "occurrences": [
            {"occurrence_id": "R1", "native_id": None, "order": 1, "native_object": native},
            {"occurrence_id": "R2", "native_id": "native-background", "order": 2, "native_object": {"note": "context"}},
        ],
        "core_occurrence_ids": ["R1"],
        "context_occurrence_ids": ["R2"],
        "unavailable_occurrence_ids": [],
        "anchors": [
            {
                "anchor_id": "A1",
                "occurrence_id": "R1",
                "pointer": "/payload/value",
                "value_type": value_type(value),
                "value": value,
                "native_object_sha256": native_object_sha256(native),
                "source_locator": "synthetic://fixture/R1/payload/value",
            }
        ],
    }
    extraction = {
        "packet_id": packet["packet_id"],
        "facts": [
            {
                "facet": "attempt_action",
                "text": "The supplied source contains this value, without proving task success.",
                "state": "observed_in_source",
                "evidence_refs": ["A1"],
                "uncertainty": ["No overall task acceptance supplied."],
            }
        ],
        "lessons": [
            {
                "kind": "conditional_advice",
                "primary_occurrence_id": "R1",
                "when_to_use": "Only within this synthetic fixture's scope.",
                "principle": "Preserve the supplied value's type and native identity.",
                "support_and_limits": "Binding alone does not prove semantic correctness.",
                "evidence_refs": ["A1"],
                "counter_evidence_refs": [],
                "alternatives": [],
                "interpretation": "proposed",
                "verification_needed": ["Independent source-supported review."],
            }
        ],
        "processed_core_ids": ["R1"],
        "unprocessed_core_ids": [],
        "omitted_findings": [],
        "complete": True,
    }
    return packet, extraction


def check(packet: dict[str, Any], extraction: dict[str, Any]) -> Any:
    return validate_bindings(canonical_json(packet), canonical_json(extraction))


@pytest.mark.parametrize("value", ['echo "x"\\\\path\nline', 4583, None, True, [1, None], {"nested": False}])
def test_typed_anchors_round_trip(value: Any) -> None:
    packet, extraction = seed_pair(value)
    restored = EvidencePacket.model_validate_json(canonical_json(packet))
    assert canonical_json(restored.anchors[0].value) == canonical_json(value)
    assert restored.occurrences[0].native_id is None
    assert check(packet, extraction).disposition == "admitted"
    assert check(packet, extraction).evidence_needed == ["Independent source-supported review."]


@pytest.mark.parametrize("provider", ["claude_code", "codex", "opencode", "unknown"])
def test_source_group_is_not_a_pilot_enum(provider: str) -> None:
    assert check(*seed_pair(provider=provider)).disposition == "admitted"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("wrong_type", "packet_contract_invalid"),
        ("wrong_hash", "packet_contract_invalid"),
        ("wrong_pointer", "packet_contract_invalid"),
        ("duplicate_occurrence", "packet_contract_invalid"),
        ("duplicate_anchor", "packet_contract_invalid"),
        ("duplicate_unavailable", "packet_contract_invalid"),
        ("overlapping_scope", "packet_contract_invalid"),
        ("future_anchor", "packet_contract_invalid"),
        ("missing_reference", "unknown_evidence_reference"),
        ("background_primary", "primary_occurrence_not_core"),
        ("wrong_packet", "packet_identity_mismatch"),
        ("unsupported_fact", "unsupported_material_fact"),
        ("unexplained_unknown", "unknown_without_evidence_need"),
        ("missing_coverage", "coverage_partition_invalid"),
        ("false_complete", "false_complete"),
    ],
)
def test_failure_regressions_are_retained(mutation: str, expected: str) -> None:
    packet, extraction = seed_pair()
    if mutation == "wrong_type":
        packet["anchors"][0]["value_type"] = "number"
    elif mutation == "wrong_hash":
        packet["anchors"][0]["native_object_sha256"] = "0" * 64
    elif mutation == "wrong_pointer":
        packet["anchors"][0]["pointer"] = "/does/not/exist"
    elif mutation == "duplicate_occurrence":
        packet["occurrences"].append(deepcopy(packet["occurrences"][0]))
    elif mutation == "duplicate_anchor":
        packet["anchors"].append(deepcopy(packet["anchors"][0]))
    elif mutation == "duplicate_unavailable":
        packet["unavailable_occurrence_ids"] = ["missing", "missing"]
    elif mutation == "overlapping_scope":
        packet["context_occurrence_ids"].append("R1")
    elif mutation == "future_anchor":
        packet["cutoff_order"] = 0
    elif mutation == "missing_reference":
        extraction["facts"][0]["evidence_refs"] = ["invented"]
    elif mutation == "background_primary":
        extraction["lessons"][0]["primary_occurrence_id"] = "R2"
    elif mutation == "wrong_packet":
        extraction["packet_id"] = "other"
    elif mutation == "unsupported_fact":
        extraction["facts"][0]["evidence_refs"] = []
    elif mutation == "unexplained_unknown":
        extraction["facts"][0]["state"] = "unknown"
        extraction["facts"][0]["uncertainty"] = []
    elif mutation == "missing_coverage":
        extraction["processed_core_ids"] = []
    elif mutation == "false_complete":
        extraction["omitted_findings"] = ["Output budget reached."]
    result = check(packet, extraction)
    assert result.disposition == "excluded"
    assert result.reasons == [expected]


@pytest.mark.parametrize("sensitivity", ["held", "unknown"])
def test_sensitivity_is_undecidable_not_an_automatic_export(sensitivity: str) -> None:
    packet, extraction = seed_pair()
    packet["sensitivity"] = sensitivity
    assert check(packet, extraction).disposition == "undecidable"


def test_partial_empty_and_unknown_outputs_are_distinct() -> None:
    packet, extraction = seed_pair()
    extraction["complete"] = False
    extraction["processed_core_ids"] = []
    extraction["unprocessed_core_ids"] = ["R1"]
    assert check(packet, extraction).disposition == "undecidable"
    extraction["complete"] = True
    extraction["processed_core_ids"] = ["R1"]
    extraction["unprocessed_core_ids"] = []
    extraction["facts"] = []
    extraction["lessons"] = []
    assert check(packet, extraction).disposition == "admitted"
    extraction["facts"] = [
        {
            "facet": "outcome",
            "text": "Unknown.",
            "state": "unknown",
            "evidence_refs": [],
            "uncertainty": ["Need final task receipt."],
        }
    ]
    assert check(packet, extraction).disposition == "admitted"


def test_json_pointer_escaping_and_no_execution() -> None:
    source = {"a/b": {"~key": ["do-not-execute"]}}
    assert json_pointer(source, "/a~1b/~0key/0") == "do-not-execute"
    with pytest.raises(ValueError, match="invalid_json_pointer"):
        json_pointer(source, "/~2")


def test_validation_errors_do_not_echo_source_values() -> None:
    packet, extraction = seed_pair("synthetic-sensitive-marker")
    packet["anchors"][0]["value"] = "wrong"
    assert "synthetic-sensitive-marker" not in check(packet, extraction).model_dump_json()
    assert validate_bindings("{broken", "{}").reasons == ["packet_contract_invalid"]
    assert validate_bindings(canonical_json(seed_pair()[0]), "{broken").reasons == ["output_contract_invalid"]


@pytest.mark.parametrize("provider", ["claude_code", "codex", "opencode"])
def test_automatic_seed_anchors_all_values_not_human_selected(provider: str) -> None:
    native = {"a/b": {"~key": ["same", 4583, None, False, [], {}]}}
    occurrences = [
        Occurrence(occurrence_id=name, native_id=None, order=index, native_object=deepcopy(native))
        for index, name in enumerate(["first-execution", "second-execution"])
    ]
    packet = packet_from_occurrences(
        packet_id="DEMO-automatic",
        provider=provider,
        source_group="unrestricted-source-group",
        sensitivity="safe",
        occurrences=occurrences,
        source_locators={row.occurrence_id: f"synthetic://source/{row.order}" for row in occurrences},
        core_occurrence_ids=[row.occurrence_id for row in occurrences],
        context_occurrence_ids=[],
        cutoff_order=2,
    )
    assert len(packet.anchors) == 12
    assert len({anchor.anchor_id for anchor in packet.anchors}) == 12
    assert all(row.native_id is None for row in packet.occurrences)
    assert {anchor.value_type for anchor in packet.anchors} == {
        "string",
        "number",
        "null",
        "boolean",
        "array",
        "object",
    }
    assert all(anchor.pointer.startswith("/a~1b/~0key/") for anchor in packet.anchors)
    restored = EvidencePacket.model_validate_json(packet.model_dump_json())
    assert restored.model_dump() == packet.model_dump()


def test_unavailable_and_present_identity_cannot_overlap() -> None:
    packet, extraction = seed_pair()
    packet["unavailable_occurrence_ids"] = ["R1"]
    assert check(packet, extraction).reasons == ["packet_contract_invalid"]


@pytest.mark.parametrize("encoded", [False, True])
def test_native_generated_dictionary_binding(encoded: bool) -> None:
    packet, extraction = seed_pair(4583)
    data = lazy.pd.DataFrame(
        [{"packet_json": canonical_json(packet), "extraction": canonical_json(extraction) if encoded else extraction}]
    )
    result = validate_generated_dataframe(data)
    assert result.iloc[0]["disposition"] == "admitted"
    assert result.iloc[0]["reasons"] == ["bindings_valid_content_not_reviewed"]


def test_native_generation_declaration_retains_original_source(tmp_path: Path) -> None:
    packet, _ = seed_pair()
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps({"packet_json": canonical_json(packet)}) + "\n")
    model = dd.ModelConfig(alias="flash", model="deepseek-flash", provider="deepseek")
    config = generation_config(seed, model).build()
    structured, validation, judge = config.columns
    assert structured.column_type == "llm-structured"
    assert validation.column_type == "validation"
    assert "binding_validation" in judge.prompt
    assert judge.column_type == "llm-judge"
    assert "packet_json" in judge.prompt
    assert "extraction" in judge.prompt
    assert "not merely a generated digest" in judge.system_prompt


def test_real_native_pipeline_all_three_providers_no_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_http(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Offline native pipeline attempted HTTP.")

    monkeypatch.setattr(httpx.Client, "send", no_http)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_http)
    rows = []
    for provider, value in [("claude_code", 'quoted "cmd"\\path'), ("codex", 4583), ("opencode", None)]:
        packet, extraction = seed_pair(value, provider)
        rows.append({"packet_json": canonical_json(packet), "extraction_json": canonical_json(extraction)})
    for mode in ["missing_reference", "background_primary", "partial"]:
        packet, extraction = seed_pair()
        packet["packet_id"] = mode
        extraction["packet_id"] = mode
        if mode == "missing_reference":
            extraction["facts"][0]["evidence_refs"] = ["invented"]
        elif mode == "background_primary":
            extraction["lessons"][0]["primary_occurrence_id"] = "R2"
        else:
            extraction["complete"] = False
            extraction["processed_core_ids"] = []
            extraction["unprocessed_core_ids"] = ["R1"]
        rows.append({"packet_json": canonical_json(packet), "extraction_json": canonical_json(extraction)})
    seed = tmp_path / "all-synthetic-seeds.jsonl"
    seed.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[dd.ModelProvider(name="offline-unused", endpoint="http://127.0.0.1:9/v1")],
    ).create(offline_config(seed), num_records=len(rows), dataset_name="offline_evidence_binding")
    frame = result.load_dataset()
    assert len(frame) == len(rows)
    assert frame["binding_disposition"].value_counts().to_dict() == {"admitted": 3, "excluded": 2, "undecidable": 1}
    assert not frame["training_admitted"].any()
    assert set(frame["content_review_status"]) == {"not_run"}
    # Round-trip original bytes and typed objects; all rows, including failures, survive.
    assert sorted(frame["packet_json"].tolist()) == sorted(row["packet_json"] for row in rows)

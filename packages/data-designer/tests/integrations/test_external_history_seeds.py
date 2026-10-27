# SPDX-FileCopyrightText: Copyright (c) 2026 DataDesigner contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
from pytest_httpx import HTTPXMock

import data_designer.config as dd
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.types import ChatCompletionRequest
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage
from data_designer.integrations.external_history_calls import (
    CallBlocked,
    CallPolicy,
    CallReceipts,
    recorded_model_registry,
)
from data_designer.integrations.external_history_seeds import (
    BINDING_PREFIX,
    DerivedInputs,
    ReaderViewStore,
    SeedOccurrence,
    binding_from_messages,
    bound_prompt,
    export_seeds,
    file_sha256,
    object_sha256,
    reader_payload_sha256,
    seed_manifest_config,
    validate_seed_manifest,
    work_binding,
)
from data_designer.interface import DataDesigner


@pytest.fixture(autouse=True)
def no_vendor_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("data_designer.engine.models.telemetry.TELEMETRY_ENABLED", False)
    monkeypatch.setattr("data_designer.engine.models.facade.TELEMETRY_ENABLED", False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    )


def fixture_inputs(tmp_path: Path, *, held: bool = True, mismatch: bool = False) -> DerivedInputs:
    atoms, index, inventory = [], [], []
    for number, provider in enumerate(["claude", "codex", "opencode"]):
        path = f"not-opened-raw/{provider}.jsonl"
        locator = (
            {"path": path, "line": number + 1, "byte_offset": 0, "offset_space": "file_utf8"}
            if provider != "opencode"
            else {"path": path, "table": "part", "row_id": "part-1", "message_id": "message-1"}
        )
        payload = {"z": 4583, "a": None, "flag": True, "list": [], "quote": 'line\\n"quoted"'}
        atom = {
            "atom_id": f"atom-{number}",
            "provider": provider,
            "session_key": f"{provider}|source|session",
            "source_namespace": f"{provider}|source",
            "source_session_id": None,
            "native_message_id": None,
            "native_role": "tool",
            "timestamp": None,
            "record_type": "test",
            "record_kind": "message",
            "origin": "tool_result",
            "locator": locator,
            "payload_sha256": reader_payload_sha256(payload),
            "raw_record_sha256": "a" * 64,
            "tool_links": [{"kind": "result", "id": None}],
            "payload": payload,
            "status": "admitted_observation",
        }
        meta = {k: v for k, v in atom.items() if k not in ["payload", "status"]}
        meta.update(
            reader_emission_ordinal=number, fixture_cohort=False, learning_analysis_disposition="admitted_analysis"
        )
        if held and number == 2:
            meta["learning_analysis_disposition"] = "undecidable_sensitive_atom"
            atom["payload"] = {"sensitive-marker": "not-export-this-body"}
        if mismatch and number == 0:
            atom["payload_sha256"] = "f" * 64
        atoms.append(atom)
        index.append(meta)
        inventory.append({"path": path, "sha256": "b" * 64, "status": "admitted_reader", "bytes": 10})
    paths = [tmp_path / name for name in ["atoms.jsonl", "index.jsonl", "input-inventory.jsonl"]]
    for path, rows in zip(paths, [atoms, index, inventory], strict=True):
        write_rows(path, rows)
    return DerivedInputs(paths[1], paths[0], paths[2], {str(p): file_sha256(p) for p in paths}, "c" * 64)


def seed_rows(path: Path) -> list[SeedOccurrence]:
    return [SeedOccurrence.model_validate_json(json.loads(line)["seed_json"]) for line in path.read_text().splitlines()]


def test_all_sources_and_sensitive_gaps_are_registered(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path)
    report = export_seeds(inputs, tmp_path / "out")
    assert (report["admitted"], report["excluded"], report["undecidable"]) == (2, 0, 1)
    assert report["registered_source_union"] == report["seed_rows"] == 3
    assert report["providers"] == {"claude_code": 1, "codex": 1, "opencode": 1}
    assert report["source_groups"] == 3 and report["join_gap_entries"] == 0
    assert "not-export-this-body" not in (tmp_path / "out/seeds.jsonl").read_text()
    rows = seed_rows(tmp_path / "out/seeds.jsonl")
    assert all(row.native_message_id is None for row in rows)
    assert rows[2].locator["table"] == "part"
    store = ReaderViewStore(inputs.atoms, inputs.expected_hashes[str(inputs.atoms)])
    try:
        assert store.resolve(rows[0]) == {"z": 4583, "a": None, "flag": True, "list": [], "quote": 'line\\n"quoted"'}
        with pytest.raises(ValueError, match="held_or_unresolved"):
            store.resolve(rows[2])
    finally:
        store.close()
    assert all(row.payload_view == "cleaned_reader_payload_not_full_native_record" for row in rows)


def test_legacy_payload_hash_is_not_canonical_hash() -> None:
    payload = {"z": 4583, "a": None}
    assert reader_payload_sha256(payload) != object_sha256(payload)


def test_all_hash_mismatches_are_unknown_not_repaired(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path, held=False, mismatch=True)
    report = export_seeds(inputs, tmp_path / "out")
    assert report["admitted"] == 2 and report["undecidable"] == 1
    assert report["reason_counts"]["reader_index_mismatch:payload_sha256"] == 1


def test_input_change_stops_before_export(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path)
    inputs.atoms.write_text(inputs.atoms.read_text() + "\n")
    with pytest.raises(ValueError, match="input_changed_before_export"):
        export_seeds(inputs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicate_source_execution_identity_is_not_content_deduplicated(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path, held=False)
    records = [json.loads(line) for line in inputs.index.read_text().splitlines()]
    records.append(records[0])
    write_rows(inputs.index, records)
    inputs.expected_hashes[str(inputs.index)] = file_sha256(inputs.index)
    with pytest.raises(ValueError, match="duplicate_index_atom_identity"):
        export_seeds(inputs, tmp_path / "out")


@pytest.mark.parametrize("missing", ["reader", "index"])
def test_join_gaps_count_as_unknown_even_without_seed_row(tmp_path: Path, missing: str) -> None:
    inputs = fixture_inputs(tmp_path, held=False)
    p = inputs.atoms if missing == "reader" else inputs.index
    records = [json.loads(line) for line in p.read_text().splitlines()][1:]
    write_rows(p, records)
    inputs.expected_hashes[str(p)] = file_sha256(p)
    report = export_seeds(inputs, tmp_path / "out")
    assert report["registered_source_union"] == 3 and report["seed_rows"] == 2
    assert report["admitted"] == 2 and report["undecidable"] == 1 and report["join_gap_entries"] == 1


def test_stable_source_identity_separate_from_revision_and_attempt(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path, held=False)
    export_seeds(inputs, tmp_path / "out1")
    export_seeds(inputs, tmp_path / "out2")
    rows1 = seed_rows(tmp_path / "out1/seeds.jsonl")
    rows2 = seed_rows(tmp_path / "out2/seeds.jsonl")
    assert rows1 == rows2
    assert len({row.occurrence_id for row in rows1}) == 3  # identical text, distinct source executions
    changed = rows1[0].model_dump(mode="json")
    changed["tool_links"] = []
    with pytest.raises(ValueError, match="seed_revision_mismatch"):
        SeedOccurrence.model_validate(changed)


def test_native_metadata_validation_keeps_unknown(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path)
    export_seeds(inputs, tmp_path / "out")
    frame = pd.read_json(tmp_path / "out/seeds.jsonl", lines=True)
    result = validate_seed_manifest(frame).to_dict(orient="records")
    assert [r["disposition"] for r in result] == ["admitted", "admitted", "undecidable"]
    assert validate_seed_manifest(pd.DataFrame({"seed_json": ["{}"]})).iloc[0]["disposition"] == "undecidable"


def test_native_sdk_preserves_every_seed_and_hold_without_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = fixture_inputs(tmp_path)
    export_seeds(inputs, tmp_path / "out")

    def no_http(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Offline seed validation attempted HTTP")

    monkeypatch.setattr(httpx.Client, "send", no_http)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_http)
    designer = DataDesigner(
        artifact_path=tmp_path / "native",
        model_providers=[dd.ModelProvider(name="unused", endpoint="http://127.0.0.1:9/v1")],
    )
    designer.set_run_config(dd.RunConfig(otel_metrics_port=None, display_tui=False))
    result = designer.create(
        seed_manifest_config(tmp_path / "out/seeds.jsonl"), num_records=3, dataset_name="seed_roundtrip"
    )
    frame = result.load_dataset()
    assert frame["seed_json"].tolist() == pd.read_json(tmp_path / "out/seeds.jsonl", lines=True)["seed_json"].tolist()
    assert [r["disposition"] for r in frame["seed_validation"]] == ["admitted", "admitted", "undecidable"]


def test_work_binding_outer_envelope_only() -> None:
    binding = work_binding("O-" + "1" * 64, "2" * 64, "extract", "3" * 64)
    prompt = bound_prompt(binding, {"text": json.dumps({"data_designer_work": {"invented": "nested"}})})
    assert binding_from_messages(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    ) == binding.model_dump(mode="json")
    assert binding_from_messages([{"role": "user", "content": "quote " + prompt}]) is None
    assert (
        binding_from_messages([{"role": "user", "content": "ordinary history"}, {"role": "tool", "content": prompt}])
        is None
    )
    bad = json.loads(prompt[len(BINDING_PREFIX) :])
    bad["data_designer_work"]["work_id"] = "W-" + "0" * 64
    with pytest.raises(ValueError):
        binding_from_messages([{"role": "user", "content": BINDING_PREFIX + json.dumps(bad)}])


def test_invalid_binding_prevents_dispatch(tmp_path: Path) -> None:
    policy = CallPolicy(
        provider_name="synthetic",
        model_id="synthetic-model",
        endpoint="https://synthetic.invalid/v1",
        price_ref="synthetic",
        cap_usd="1",
        input_usd_per_million="1",
        output_usd_per_million="1",
        max_input_tokens=100,
        max_output_tokens=100,
    )
    with CallReceipts(tmp_path / "calls.jsonl", policy) as receipts:
        with pytest.raises(CallBlocked, match="invalid_work_binding"):
            receipts.begin(
                ChatCompletionRequest(
                    model=policy.model_id,
                    messages=[{"role": "user", "content": BINDING_PREFIX + '{"data_designer_work":{}}'}],
                    max_tokens=10,
                )
            )
        assert receipts.summary()["excluded_with_reason"] == {"invalid_work_binding": 1}


def test_native_columns_bind_source_work_to_distinct_http_attempts(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    bindings = [work_binding("O-" + char * 64, "2" * 64, "extract", "3" * 64) for char in ["1", "4"]]
    seed = tmp_path / "bound-seeds.jsonl"
    write_rows(
        seed,
        [
            {"bound_prompt": bound_prompt(binding, {"same_text": "same content is not the same execution"})}
            for binding in bindings
        ],
    )
    model = dd.ModelConfig(
        alias="synthetic",
        model="synthetic-model",
        provider="synthetic",
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(max_tokens=10),
    )
    policy = CallPolicy(
        provider_name="synthetic",
        model_id=model.model,
        endpoint="https://synthetic.invalid/v1",
        price_ref="synthetic",
        cap_usd="1",
        input_usd_per_million="1",
        output_usd_per_million="1",
        max_input_tokens=100,
        max_output_tokens=100,
    )
    httpx_mock.add_response(
        json={
            "choices": [{"message": {"role": "assistant", "content": "first"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
        is_reusable=True,
    )
    with CallReceipts(tmp_path / "calls.jsonl", policy) as receipts:
        registry = recorded_model_registry(
            model_configs=[model],
            secret_resolver=MagicMock(spec=SecretResolver),
            model_provider_registry=ModelProviderRegistry(
                providers=[dd.ModelProvider(name="synthetic", endpoint=policy.endpoint)]
            ),
            receipts=receipts,
            client_concurrency_mode=ClientConcurrencyMode.ASYNC,
        )
        cfg = dd.DataDesignerConfigBuilder(model_configs=[model])
        cfg.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed)))
        cfg.add_column(dd.LLMTextColumnConfig(name="answer", model_alias=model.alias, prompt="{{bound_prompt}}"))
        (tmp_path / "native").mkdir()
        provider = ResourceProvider(
            artifact_storage=ArtifactStorage(artifact_path=tmp_path / "native"),
            model_registry=registry,
            run_config=dd.RunConfig(otel_metrics_port=None),
        )
        reader = DataFrameSeedReader()
        reader.attach(dd.DataFrameSeedSource(df=pd.read_json(seed, lines=True)), MagicMock(spec=SecretResolver))
        provider.seed_reader = reader
        try:
            DatasetBuilder(data_designer_config=cfg.build(), resource_provider=provider).build(num_records=2)
        finally:
            asyncio.run(registry.aclose())
    reserved = [
        json.loads(line)
        for line in (tmp_path / "calls.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "reserved"
    ]
    assert {entry["declared_work_binding"]["work_id"] for entry in reserved} == {
        binding.work_id for binding in bindings
    }
    assert len({entry["request_id"] for entry in reserved}) == 2
    assert len(httpx_mock.get_requests()) == 2


def test_original_reader_sorted_writer_cannot_replay_insertion_hash(tmp_path: Path) -> None:
    original = {"z": 4583, "a": None}
    old_hash = reader_payload_sha256(original)
    persisted = json.loads(json.dumps({"payload": original}, sort_keys=True))["payload"]
    assert reader_payload_sha256(persisted) != old_hash
    assert object_sha256(persisted) == object_sha256(original)
    inputs = fixture_inputs(tmp_path, held=False)
    report = export_seeds(inputs, tmp_path / "out")
    assert report["admitted"] == 3
    assert report["legacy_hash_replay_counts"] == {"unreplayable_after_sorted_serialization": 3}
    assert all(
        row.payload_sha256_recorded != row.payload_view_sha256 for row in seed_rows(tmp_path / "out/seeds.jsonl")
    )


def test_sealed_reader_body_change_cannot_be_silently_accepted(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path, held=False)
    export_seeds(inputs, tmp_path / "out")
    old_sha = inputs.expected_hashes[str(inputs.atoms)]
    inputs.atoms.write_text(inputs.atoms.read_text().replace("4583", "9999"))
    with pytest.raises(ValueError, match="reader_artifact_hash_mismatch"):
        ReaderViewStore(inputs.atoms, old_sha)


@pytest.mark.parametrize("body", ["not-json", "{}", "[]"])
def test_declared_but_malformed_envelope_is_not_silently_unbound(body: str) -> None:
    with pytest.raises(ValueError):
        binding_from_messages([{"role": "user", "content": BINDING_PREFIX + body}])


def test_same_work_has_distinct_attempts_not_content_deduplication(tmp_path: Path) -> None:
    binding = work_binding("O-" + "1" * 64, "2" * 64, "extract", "3" * 64)
    policy = CallPolicy(
        provider_name="synthetic",
        model_id="synthetic-model",
        endpoint="https://synthetic.invalid/v1",
        price_ref="synthetic",
        cap_usd="1",
        input_usd_per_million="1",
        output_usd_per_million="1",
        max_input_tokens=100,
        max_output_tokens=100,
    )
    request = ChatCompletionRequest(
        model=policy.model_id,
        messages=[{"role": "user", "content": bound_prompt(binding, {"fixture": "same work"})}],
        max_tokens=10,
    )
    with CallReceipts(tmp_path / "calls.jsonl", policy) as receipts:
        first = receipts.begin(request)
        second = receipts.begin(request)
        assert first != second
    calls = [
        json.loads(line)
        for line in (tmp_path / "calls.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "reserved"
    ]
    assert len(calls) == 2 and {row["declared_work_binding"]["work_id"] for row in calls} == {binding.work_id}

# SPDX-FileCopyrightText: Copyright (c) 2026 DataDesigner contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import data_designer.config as dd
from data_designer.config.run_config import ResumeMode
from data_designer.engine.models.clients.retry import RetryConfig, create_retry_transport
from data_designer.integrations.external_history import canonical_json, offline_config
from data_designer.interface import DataDesigner


@pytest.mark.parametrize(("retries", "attempts"), [(0, 1), (3, 4)])
def test_native_post_receipt_loss_retry_boundary(retries: int, attempts: int) -> None:
    observed = []

    def accepted_then_lost(request: httpx.Request) -> httpx.Response:
        observed.append(request.method)
        raise httpx.ReadTimeout("Synthetic response lost after acceptance.", request=request)

    transport = create_retry_transport(
        RetryConfig(max_retries=retries, backoff_factor=0, backoff_jitter=0, max_backoff_wait=0.001),
        transport=httpx.MockTransport(accepted_then_lost),
    )
    with httpx.Client(transport=transport) as client, pytest.raises(httpx.ReadTimeout):
        client.post("https://synthetic.invalid/v1/chat/completions", json={"synthetic": True})
    assert observed == ["POST"] * attempts
    # This is the real native transport at a synthetic HTTP boundary, not a
    # claim that model-client configuration already exposes this setting.


def test_native_completed_row_group_resume_preserves_occurrence_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real NeMo executor/checkpoint, not a custom scheduler or patched builder."""

    def no_http(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Offline resume attempted HTTP.")

    monkeypatch.setattr(httpx.Client, "send", no_http)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_http)
    # Intentionally invalid input contracts still survive as excluded rows.
    # This checks occurrence/first-answer persistence, not semantic correctness.
    rows = [
        {"seed_occurrence": str(index), "packet_json": canonical_json({"fixture": index}), "extraction_json": "{}"}
        for index in range(6)
    ]
    seed = tmp_path / "all-rows.jsonl"
    seed.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = offline_config(seed)
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[dd.ModelProvider(name="offline-unused", endpoint="http://127.0.0.1:9/v1")],
    )
    first = designer.create(config, num_records=1, dataset_name="resume_occurrences")
    assert first.load_dataset()["seed_occurrence"].tolist() == ["0"]
    completed = list(first.artifact_storage.final_dataset_path.glob("*.parquet"))
    assert completed
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in completed}
    resumed = designer.create(config, num_records=6, dataset_name="resume_occurrences", resume=ResumeMode.ALWAYS)
    frame = resumed.load_dataset()
    assert frame["seed_occurrence"].tolist() == [str(index) for index in range(6)]
    assert set(frame["binding_disposition"]) == {"excluded"}
    assert not frame["training_admitted"].any()
    assert resumed.artifact_storage.base_dataset_path == first.artifact_storage.base_dataset_path
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())

# SPDX-FileCopyrightText: Copyright (c) 2026 DataDesigner contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from pytest_httpx import HTTPXMock

from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from data_designer.engine.models.errors import (
    ModelAPIConnectionError,
    ModelAPIError,
    ModelInternalServerError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.integrations.external_history_calls import (
    CallBlocked,
    CallPolicy,
    CallReceipts,
    RecordedChatClient,
    recorded_model_registry,
)


@pytest.fixture(autouse=True)
def opt_out_of_native_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    # Production uses NEMO_TELEMETRY_ENABLED=false before imports. These two
    # native flags are already imported by the suite; apply the same opt-out
    # locally without changing other tests or replacing model execution.
    monkeypatch.setattr("data_designer.engine.models.telemetry.TELEMETRY_ENABLED", False)
    monkeypatch.setattr("data_designer.engine.models.facade.TELEMETRY_ENABLED", False)


def policy(**kwargs: Any) -> CallPolicy:
    return CallPolicy(
        **{
            "provider_name": "synthetic",
            "model_id": "synthetic-model",
            "endpoint": "https://synthetic.invalid/v1",
            "price_ref": "synthetic-price-not-commercial",
            "cap_usd": "0.004",
            "input_usd_per_million": "1",
            "output_usd_per_million": "2",
            "max_input_tokens": 1000,
            "max_output_tokens": 500,
            **kwargs,
        }
    )


def request(**kwargs: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        **{
            "model": "synthetic-model",
            "messages": [{"role": "user", "content": "synthetic only"}],
            "max_tokens": 100,
            **kwargs,
        }
    )


def response(**kwargs: Any) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        **{
            "message": AssistantMessage(content="first answer"),
            "usage": Usage(input_tokens=100, output_tokens=10),
            **kwargs,
        }
    )


def payload(usage: dict[str, Any] | None = None, content: str = "first answer") -> dict[str, Any]:
    result = {
        "id": "synthetic-response",
        "object": "chat.completion",
        "model": "synthetic-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def registry(receipts: CallReceipts, *, asynchronous: bool = False, skip_health_check: bool = True) -> Any:
    return recorded_model_registry(
        model_configs=[
            ModelConfig(
                alias="synthetic-alias",
                model="synthetic-model",
                provider="synthetic",
                skip_health_check=skip_health_check,
                inference_parameters=ChatCompletionInferenceParams(max_tokens=100),
            )
        ],
        secret_resolver=MagicMock(spec=SecretResolver),
        model_provider_registry=ModelProviderRegistry(
            providers=[ModelProvider(name="synthetic", endpoint="https://synthetic.invalid/v1", provider_type="openai")]
        ),
        receipts=receipts,
        client_concurrency_mode=ClientConcurrencyMode.ASYNC if asynchronous else ClientConcurrencyMode.SYNC,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cap_usd": "NaN"},
        {"cap_usd": "-1"},
        {"cap_usd": "0"},
        {"input_usd_per_million": "Infinity"},
        {"max_input_tokens": 0},
        {"max_output_tokens": True},
        {"price_ref": ""},
        {"input_usd_per_million": "0", "output_usd_per_million": "0"},
    ],
)
def test_policy_rejects_invalid_bounds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        policy(**kwargs)


def test_used_and_inflight_are_reserved_before_dispatch(tmp_path: Path) -> None:
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        first = receipts.begin(request())
        second = receipts.begin(request())
        with pytest.raises(CallBlocked, match="cap_exhausted"):
            receipts.begin(request())
        assert Decimal(receipts.summary()["reserved_usd"]) == Decimal("0.004")
        receipts.response(first, response())
        assert Decimal(receipts.summary()["used_usd"]) == Decimal("0.00012")
        with pytest.raises(CallBlocked):
            receipts.begin(request())
        receipts.response(second, response())
        assert receipts.begin(request())
        assert receipts.summary()["excluded_with_reason"] == {"cap_exhausted": 2}


def test_concurrent_reservations_never_exceed_frozen_cap(tmp_path: Path) -> None:
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:

        def begin(_: int) -> bool:
            try:
                receipts.begin(request())
                return True
            except CallBlocked:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert sum(pool.map(begin, range(24))) == 2
        assert receipts.summary()["excluded"] == 22
        assert Decimal(receipts.summary()["reserved_usd"]) == Decimal("0.004")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": None},
        {"max_tokens": 501},
        {"max_tokens": True},
        {"model": "different"},
        {"n": 2},
        {"extra_body": {"max_tokens": 10000}},
        {"extra_body": {"stream": True}},
    ],
)
def test_invalid_requests_are_excluded_before_http(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        with pytest.raises(CallBlocked):
            receipts.begin(request(**kwargs))
        assert receipts.summary()["excluded"] == 1
        assert receipts.summary()["undecidable"] == 0


@pytest.mark.parametrize("usage", [None, Usage(input_tokens=1), Usage(input_tokens=-1, output_tokens=2)])
def test_unknown_usage_retains_reservation_and_first_response(tmp_path: Path, usage: Usage | None) -> None:
    path = tmp_path / "calls.jsonl"
    with CallReceipts(path, policy()) as receipts:
        rid = receipts.begin(request())
        receipts.response(rid, response(usage=usage))
        with pytest.raises(CallBlocked, match="unresolved_call"):
            receipts.begin(request())
        assert receipts.summary()["undecidable"] == 1
        assert Decimal(receipts.summary()["reserved_usd"]) == Decimal("0.002")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[2]["response"]["message"]["content"] == "first answer"


def test_input_ceiling_overrun_is_recorded_not_hidden(tmp_path: Path) -> None:
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        rid = receipts.begin(request())
        receipts.response(rid, response(usage=Usage(input_tokens=2000, output_tokens=10)))
        assert receipts.summary()["needs"][0]["reason"] == "token_ceiling_exceeded"
        assert Decimal(receipts.summary()["used_usd"]) == Decimal("0.00202")
        with pytest.raises(CallBlocked):
            receipts.begin(request())


def test_restart_preserves_unknown_and_never_replays_post(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    with CallReceipts(path, policy()) as receipts:
        rid = receipts.begin(request())
    original = path.read_bytes()
    with CallReceipts(path, policy()) as receipts:
        assert receipts.summary()["needs"][0]["reason"] == "interrupted_after_reservation"
        assert path.read_bytes() == original
        with pytest.raises(CallBlocked):
            receipts.begin(request())
        receipts.reconcile(
            [{"request_id": rid, "evidence_ref": "synthetic-provider-billing-receipt", "cost_usd": "0.0001"}]
        )
        assert receipts.summary()["admitted"] == 1
        assert receipts.summary()["reserved_usd"] == "0"
        assert receipts.begin(request())


def test_reconciliation_prevalidates_entire_batch(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    with CallReceipts(path, policy()) as receipts:
        rid = receipts.begin(request())
        receipts.uncertain(rid, "synthetic-lost-response")
        original = path.read_bytes()
        with pytest.raises(ValueError):
            receipts.reconcile(
                [
                    {"request_id": rid, "evidence_ref": "receipt", "cost_usd": "0.0001"},
                    {"request_id": rid, "evidence_ref": "", "cost_usd": "0"},
                ]
            )
        assert path.read_bytes() == original
        assert receipts.summary()["undecidable"] == 1


def test_policy_and_corrupt_receipts_cannot_be_silently_repaired(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    with CallReceipts(path, policy()):
        with pytest.raises(BlockingIOError):
            CallReceipts(path, policy())
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Frozen policy"):
        CallReceipts(path, policy(cap_usd="0.008"))
    assert path.read_bytes() == original
    path.write_bytes(original + b"{partial")
    with pytest.raises(ValueError):
        CallReceipts(path, policy())
    assert path.read_bytes() == original + b"{partial"


def test_header_credentials_are_not_written_to_receipts(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    with CallReceipts(path, policy()) as receipts:
        receipts.begin(request(extra_headers={"Authorization": "synthetic-secret-must-not-be-copied"}))
    assert b"synthetic-secret-must-not-be-copied" not in path.read_bytes()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("accepted_lost", ModelAPIConnectionError),
        ("timeout", ModelTimeoutError),
        ("status_500", ModelInternalServerError),
        ("status_429", ModelRateLimitError),
        ("invalid_json", ModelAPIError),
    ],
)
def test_real_model_registry_and_native_client_do_not_resend(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
    failure: str,
    expected: type[Exception],
) -> None:
    observed = []

    def fault(req: httpx.Request) -> httpx.Response:
        observed.append(req.headers["X-DataDesigner-Request-Id"])
        if failure == "accepted_lost":
            raise httpx.RemoteProtocolError("Synthetic accepted request, lost response", request=req)
        if failure == "timeout":
            raise httpx.ReadTimeout("Synthetic timeout after acceptance", request=req)
        if failure == "invalid_json":
            return httpx.Response(200, text="not-json")
        return httpx.Response(429 if failure == "status_429" else 500, json={"error": {"message": "synthetic failure"}})

    httpx_mock.add_callback(fault, is_reusable=True)
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        reg = registry(receipts)
        model = reg.get_model(model_alias="synthetic-alias")
        try:
            with pytest.raises(expected):
                model.generate("synthetic input", max_correction_steps=0, max_conversation_restarts=0)
            assert len(observed) == 1
            assert receipts.summary()["undecidable"] == 1
            with pytest.raises(CallBlocked):
                model.generate("second synthetic input")
            assert len(observed) == 1
        finally:
            reg.close()


def test_real_native_success_preserves_first_answer_usage_and_id(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json=payload({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}, "  first answer\n")
    )
    path = tmp_path / "calls.jsonl"
    with CallReceipts(path, policy()) as receipts:
        reg = registry(receipts)
        try:
            answer, _ = reg.get_model(model_alias="synthetic-alias").generate("synthetic input")
            assert answer == "first answer"  # Native facade strips; receipt must not.
            assert receipts.summary()["admitted"] == 1
            assert Decimal(receipts.summary()["used_usd"]) == Decimal("0.00012")
        finally:
            reg.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[2]["response"]["message"]["content"] == "  first answer\n"
    assert httpx_mock.get_request().headers["X-DataDesigner-Request-Id"] == events[1]["request_id"]


@pytest.mark.asyncio
async def test_real_native_async_response_loss_is_one_post(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("synthetic accepted-then-lost"), is_reusable=True)
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        reg = registry(receipts, asynchronous=True)
        try:
            with pytest.raises(ModelTimeoutError):
                await reg.get_model(model_alias="synthetic-alias").agenerate("synthetic input")
            assert len(httpx_mock.get_requests()) == 1
            assert receipts.summary()["undecidable"] == 1
        finally:
            await reg.aclose()


@pytest.mark.asyncio
async def test_async_cancellation_retains_reservation(tmp_path: Path) -> None:
    inner = MagicMock()
    inner.provider_name = "synthetic"
    entered = asyncio.Event()

    async def wait(req: ChatCompletionRequest) -> ChatCompletionResponse:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    inner.acompletion = wait
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        task = asyncio.create_task(RecordedChatClient(inner, receipts).acompletion(request()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert receipts.summary()["undecidable"] == 1
        assert Decimal(receipts.summary()["reserved_usd"]) == Decimal("0.002")


def test_frozen_model_route_is_enforced(tmp_path: Path) -> None:
    with CallReceipts(tmp_path / "calls.jsonl", policy(endpoint="https://different.invalid/v1")) as receipts:
        reg = registry(receipts)
        with pytest.raises(ValueError, match="frozen call policy"):
            reg.get_model(model_alias="synthetic-alias")


def test_second_response_does_not_modify_artifact(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    with CallReceipts(path, policy()) as receipts:
        rid = receipts.begin(request())
        receipts.response(rid, response())
        original = path.read_bytes()
        with pytest.raises(ValueError):
            receipts.response(rid, response(message=AssistantMessage(content="changed")))
        assert path.read_bytes() == original


def test_receipt_write_failure_prevents_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inner = MagicMock()
    inner.provider_name = "synthetic"
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:

        def fail(fd: int) -> None:
            raise OSError("synthetic disk failure")

        monkeypatch.setattr("os.fsync", fail)
        client = RecordedChatClient(inner, receipts)
        with pytest.raises(OSError, match="synthetic disk failure"):
            client.completion(request())
        inner.completion.assert_not_called()
        with pytest.raises(CallBlocked):
            client.completion(request())
        inner.completion.assert_not_called()


def test_unserializable_response_blocks_further_dispatch(tmp_path: Path) -> None:
    inner = MagicMock()
    inner.provider_name = "synthetic"
    inner.completion.return_value = response(raw=object())
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        client = RecordedChatClient(inner, receipts)
        with pytest.raises(TypeError):
            client.completion(request())
        with pytest.raises(CallBlocked):
            client.completion(request())
        assert inner.completion.call_count == 1
        assert receipts.summary()["undecidable"] == 1


def test_known_first_answer_survives_settlement_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    with CallReceipts(path, policy()) as receipts:
        rid = receipts.begin(request())
        receipts.response(rid, response(usage=None))
        first = path.read_bytes()
        receipts.reconcile([{"request_id": rid, "evidence_ref": "synthetic-invoice", "cost_usd": "0.0002"}])
        assert path.read_bytes().startswith(first)
    with CallReceipts(path, policy()) as receipts:
        assert receipts.summary()["admitted"] == 1
        assert Decimal(receipts.summary()["used_usd"]) == Decimal("0.0002")
    assert [json.loads(x) for x in path.read_text().splitlines()][2]["response"]["message"]["content"] == "first answer"


def test_actual_local_http_acceptance_then_disconnect_is_not_replayed(tmp_path: Path) -> None:
    import socket
    import threading
    from dataclasses import replace
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            observed.append({"request_id": self.headers["X-DataDesigner-Request-Id"], "body": json.loads(body)})
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        with CallReceipts(tmp_path / "calls.jsonl", replace(policy(), endpoint=endpoint)) as receipts:
            reg = recorded_model_registry(
                model_configs=[
                    ModelConfig(
                        alias="synthetic-alias",
                        model="synthetic-model",
                        provider="synthetic",
                        inference_parameters=ChatCompletionInferenceParams(max_tokens=100),
                    )
                ],
                secret_resolver=MagicMock(spec=SecretResolver),
                model_provider_registry=ModelProviderRegistry(
                    providers=[ModelProvider(name="synthetic", endpoint=endpoint, provider_type="openai")]
                ),
                receipts=receipts,
            )
            try:
                with pytest.raises(ModelAPIConnectionError):
                    reg.get_model(model_alias="synthetic-alias").generate("synthetic input")
                assert len(observed) == 1
                assert observed[0]["body"]["model"] == "synthetic-model"
                assert observed[0]["request_id"] == receipts.summary()["needs"][0]["request_id"]
                with pytest.raises(CallBlocked):
                    reg.get_model(model_alias="synthetic-alias").generate("not replayed")
                assert len(observed) == 1
            finally:
                reg.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.asyncio
async def test_actual_native_async_cancellation_is_unresolved(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    entered = asyncio.Event()

    async def accept(req: httpx.Request) -> httpx.Response:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    httpx_mock.add_callback(accept)
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        reg = registry(receipts, asynchronous=True)
        try:
            task = asyncio.create_task(reg.get_model(model_alias="synthetic-alias").agenerate("synthetic input"))
            await asyncio.wait_for(entered.wait(), timeout=3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(httpx_mock.get_requests()) == 1
            assert receipts.summary()["undecidable"] == 1
            assert receipts.summary()["needs"][0]["reason"] == "CancelledError"
        finally:
            await reg.aclose()


@pytest.mark.parametrize(("skip_health_check", "expected_calls"), [(True, 2), (False, 3)])
def test_native_dataset_builder_uses_recorded_registry_and_writes_parquet(
    tmp_path: Path, httpx_mock: HTTPXMock, skip_health_check: bool, expected_calls: int
) -> None:
    import pandas as pd

    import data_designer.config as dd
    from data_designer.config.run_config import RunConfig
    from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
    from data_designer.engine.resources.resource_provider import ResourceProvider
    from data_designer.engine.storage.artifact_storage import ArtifactStorage

    httpx_mock.add_response(
        json=payload({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}), is_reusable=True
    )
    with CallReceipts(tmp_path / "calls.jsonl", policy()) as receipts:
        reg = registry(receipts, asynchronous=True, skip_health_check=skip_health_check)
        configs = list(reg.model_configs.values())
        config = dd.DataDesignerConfigBuilder(model_configs=configs)
        config.add_column(
            dd.LLMTextColumnConfig(name="answer", model_alias="synthetic-alias", prompt="synthetic input")
        )
        provider = ResourceProvider(
            artifact_storage=ArtifactStorage(artifact_path=tmp_path, dataset_name="native"),
            model_registry=reg,
            run_config=RunConfig(otel_metrics_port=None),
        )
        builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=provider)
        try:
            result = builder.build(num_records=2)
            assert pd.read_parquet(result)["answer"].tolist() == ["first answer", "first answer"]
            assert receipts.summary()["admitted"] == expected_calls
            assert receipts.summary()["training_admitted"] == 0
            assert len(httpx_mock.get_requests()) == expected_calls
        finally:
            asyncio.run(reg.aclose())


def test_native_error_message_is_preserved_except_known_credentials(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    inner = MagicMock()
    inner.provider_name = "synthetic"
    inner.completion.side_effect = RuntimeError("synthetic failure; key=secret-key header=header-value")
    with CallReceipts(path, policy()) as receipts:
        client = RecordedChatClient(inner, receipts, credential_values=("secret-key",))
        with pytest.raises(RuntimeError):
            client.completion(request(extra_headers={"Authorization": "header-value"}))
    data = path.read_text()
    assert "synthetic failure" in data
    assert "secret-key" not in data
    assert "header-value" not in data
    assert json.loads(data.splitlines()[2])["error"]["credential_redactions"] == 2

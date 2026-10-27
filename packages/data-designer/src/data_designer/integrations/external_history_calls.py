# SPDX-FileCopyrightText: Copyright (c) 2026 DataDesigner contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in chat-call receipts and reservations, using native NeMo clients.

This is a POSIX, single-writer receipt artifact, not an execution engine or a
content truth store. Token ceilings and prices are operator-supplied assumptions.
It does not establish exactly-once provider execution or permission to send data.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Self

from data_designer.config.models import ModelConfig
from data_designer.engine.errors import DataDesignerError
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.factory import create_model_client
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.models.request_admission.controller import AdaptiveRequestAdmissionController
from data_designer.engine.secret_resolver import SecretResolver


class CallBlocked(DataDesignerError):
    """The receipt boundary stopped a call before dispatch."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _money(value: str | Decimal) -> Decimal:
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError("Money must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class CallPolicy:
    provider_name: str
    model_id: str
    endpoint: str
    price_ref: str
    cap_usd: str
    input_usd_per_million: str
    output_usd_per_million: str
    max_input_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        for text in (self.provider_name, self.model_id, self.endpoint, self.price_ref):
            if not text:
                raise ValueError("Policy identity and price reference are required")
        for count in (self.max_input_tokens, self.max_output_tokens):
            if type(count) is not int or count <= 0:
                raise ValueError("Token ceilings must be positive integers")
        for amount in (self.cap_usd, self.input_usd_per_million, self.output_usd_per_million):
            _money(amount)
        if self.reservation_usd <= 0 or _money(self.cap_usd) <= 0:
            raise ValueError("A positive cap and reservation are required")

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            Decimal(input_tokens) * _money(self.input_usd_per_million)
            + Decimal(output_tokens) * _money(self.output_usd_per_million)
        ) / Decimal(1_000_000)

    @property
    def reservation_usd(self) -> Decimal:
        return self.cost(self.max_input_tokens, self.max_output_tokens)


class CallReceipts:
    """Append-only, fsynced calls; interrupted reservations stay unresolved.

    Hold this object for the run lifetime. One process owns the file; threads
    share its lock. Reopening never resends a request. Preserve this artifact
    alongside NeMo artifacts, with the same privacy/retention policy as inputs.
    """

    def __init__(self, path: Path, policy: CallPolicy) -> None:
        import fcntl

        self.path = path
        self.policy = policy
        self._mutex = threading.RLock()
        self._closed = False
        self._broken = False
        self._events: list[dict[str, Any]] = []
        self._calls: dict[str, dict[str, Any]] = {}
        self._rejections: Counter[str] = Counter()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock_file = path.with_suffix(path.suffix + ".lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._file = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600), "a+")
            self._file.seek(0)
            for line in self._file:
                event = json.loads(line)
                digest = event.pop("sha256")
                if (
                    event["previous_sha256"] != self._previous_hash()
                    or digest != hashlib.sha256(_json(event).encode()).hexdigest()
                ):
                    raise ValueError("Receipt integrity mismatch; original bytes retained")
                event["sha256"] = digest
                self._apply(event)
            if self._events:
                if self._events[0].get("policy") != asdict(policy):
                    raise ValueError("Frozen policy mismatch; use a new receipt artifact")
                for call in self._calls.values():
                    if call["state"] == "in_flight":
                        call["state"] = "undecidable"
                        call["reason"] = "interrupted_after_reservation"
            else:
                self._append({"kind": "policy", "policy": asdict(policy)})
        except BaseException:
            if hasattr(self, "_file"):
                self._file.close()
            self._lock_file.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._mutex:
            if not self._closed:
                self._file.close()
                self._lock_file.close()
                self._closed = True

    def _previous_hash(self) -> str:
        return self._events[-1]["sha256"] if self._events else "0" * 64

    def _append(self, event: dict[str, Any]) -> None:
        if self._closed or self._broken:
            raise CallBlocked("Receipt artifact is closed or has a write failure")
        try:
            event = {**event, "utc": datetime.now(UTC).isoformat(), "previous_sha256": self._previous_hash()}
            event["sha256"] = hashlib.sha256(_json(event).encode()).hexdigest()
            self._file.write(_json(event) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())
        except BaseException:
            self._broken = True
            raise
        self._apply(event)

    def _apply(self, event: dict[str, Any]) -> None:
        kind = event["kind"]
        if kind == "reserved":
            if event["request_id"] in self._calls:
                raise ValueError("Duplicate request identity")
            self._calls[event["request_id"]] = {"state": "in_flight", "reason": "in_flight", "cost": None}
        elif kind == "rejected":
            self._rejections[event["reason"]] += 1
        elif kind in ("response", "uncertain", "reconciled"):
            call = self._calls[event["request_id"]]
            if call["state"] == "admitted":
                raise ValueError("A settled call cannot be overwritten")
            if kind == "response":
                if call.get("response_saved"):
                    raise ValueError("First response cannot be overwritten")
                call["response_saved"] = True
            call.update(state=event["state"], reason=event["reason"], cost=event.get("cost_usd"))
        elif kind != "policy":
            raise ValueError("Unknown receipt event")
        self._events.append(event)

    def _amounts(self) -> tuple[Decimal, Decimal]:
        used = sum((_money(c["cost"]) for c in self._calls.values() if c["cost"] is not None), Decimal(0))
        reserved = sum((self.policy.reservation_usd for c in self._calls.values() if c["cost"] is None), Decimal(0))
        return used, reserved

    def begin(self, request: ChatCompletionRequest) -> str:
        with self._mutex:
            request_id = uuid.uuid4().hex
            body = asdict(request)
            body.pop("extra_headers", None)  # Authorization/custom headers never enter receipts.
            reason = None
            if request.model != self.policy.model_id:
                reason = "model_mismatch"
            elif type(request.max_tokens) is not int or not 0 < request.max_tokens <= self.policy.max_output_tokens:
                reason = "output_limit_missing_or_exceeded"
            elif (request.n is not None and (type(request.n) is not int or request.n != 1)) or any(
                key in (request.extra_body or {})
                for key in ("model", "max_tokens", "max_completion_tokens", "n", "stream", "messages", "tools")
            ):
                reason = "unsupported_request_override"
            elif any(c["state"] == "undecidable" for c in self._calls.values()):
                reason = "unresolved_call"
            used, reserved = self._amounts()
            if reason is None and used + reserved + self.policy.reservation_usd > _money(self.policy.cap_usd):
                reason = "cap_exhausted"
            if reason is not None:
                self._append({"kind": "rejected", "request_id": request_id, "reason": reason})
                raise CallBlocked(reason)
            self._append({"kind": "reserved", "request_id": request_id, "request": body})
            return request_id

    def uncertain(self, request_id: str, reason: str, *, error: dict[str, Any] | None = None) -> None:
        with self._mutex:
            if self._calls[request_id]["state"] != "in_flight":
                raise ValueError("Only an in-flight call can become uncertain")
            self._append(
                {
                    "kind": "uncertain",
                    "request_id": request_id,
                    "state": "undecidable",
                    "reason": reason,
                    "error": error,
                }
            )

    def response(self, request_id: str, response: ChatCompletionResponse) -> None:
        with self._mutex:
            if self._calls[request_id]["state"] != "in_flight":
                raise ValueError("First response cannot be overwritten")
            usage = response.usage
            counts = (usage.input_tokens, usage.output_tokens) if usage is not None else (None, None)
            reason = "usage_missing_or_invalid"
            cost = None
            if all(type(n) is int and n >= 0 for n in counts):
                cost = str(self.policy.cost(*counts))
                reason = "receipt_complete"
                if counts[0] > self.policy.max_input_tokens or counts[1] > self.policy.max_output_tokens:
                    reason = "token_ceiling_exceeded"
            self._append(
                {
                    "kind": "response",
                    "request_id": request_id,
                    "state": "admitted" if reason == "receipt_complete" else "undecidable",
                    "reason": reason,
                    "cost_usd": cost,
                    "response": asdict(response),
                }
            )

    def reconcile(self, settlements: list[dict[str, str]]) -> None:
        """Append provider-evidence settlements as one batch; never replay POSTs.

        Evidence references are operator assertions, not automatically verified
        provider evidence or semantic approval. Original response bytes remain.
        """
        with self._mutex:
            ids = [entry["request_id"] for entry in settlements]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate reconciliation identity")
            for entry in settlements:
                if self._calls[entry["request_id"]]["state"] != "undecidable" or not entry["evidence_ref"].strip():
                    raise ValueError("Reconciliation requires an unknown call and an evidence reference")
                _money(entry["cost_usd"])
            for entry in settlements:
                self._append(
                    {**entry, "kind": "reconciled", "state": "admitted", "reason": "operator_evidence_settlement"}
                )

    def summary(self) -> dict[str, Any]:
        with self._mutex:
            used, reserved = self._amounts()
            counts = Counter(c["state"] for c in self._calls.values())
            return {
                "unit": "call_receipts_not_content_or_training_samples",
                "admission_definition": "cost-accounted calls; a returned answer is not implied",
                "responses_saved": sum(bool(c.get("response_saved")) for c in self._calls.values()),
                "admitted": counts["admitted"],
                "excluded_with_reason": dict(self._rejections),
                "excluded": sum(self._rejections.values()),
                "undecidable": counts["undecidable"] + counts["in_flight"],
                "used_usd": str(used),
                "reserved_usd": str(reserved),
                "cap_usd": self.policy.cap_usd,
                "needs": [
                    {
                        "request_id": rid,
                        "reason": c["reason"],
                        "evidence_needed": "Provider usage/billing or non-acceptance receipt tied to this request ID; no automatic resend",
                    }
                    for rid, c in self._calls.items()
                    if c["state"] != "admitted"
                ],
                "training_admitted": 0,
            }


class RecordedChatClient(ModelClient):
    """Record the native client's first answer; do not retry or repair it.

    Construct via ``recorded_model_registry`` so both native retry layers are
    disabled. Wrapping an arbitrary retrying client does not make it no-retry.
    """

    def __init__(self, inner: ModelClient, receipts: CallReceipts, *, credential_values: tuple[str, ...] = ()) -> None:
        self._inner = inner
        self.receipts = receipts
        self._credential_values = credential_values
        if inner.provider_name != receipts.policy.provider_name:
            raise ValueError("Provider identity mismatch")

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    def supports_chat_completion(self) -> bool:
        return self._inner.supports_chat_completion()

    def supports_embeddings(self) -> bool:
        return False

    def supports_image_generation(self) -> bool:
        return False

    def close(self) -> None:
        self._inner.close()

    async def aclose(self) -> None:
        await self._inner.aclose()

    def _error_evidence(self, exc: BaseException, request: ChatCompletionRequest) -> dict[str, Any]:
        message = str(exc)
        redactions = 0
        for value in (*self._credential_values, *(request.extra_headers or {}).values()):
            if value and value in message:
                redactions += message.count(value)
                message = message.replace(value, "[REDACTED_HEADER_OR_CREDENTIAL]")
        return {
            "class": type(exc).__name__,
            "message": message,
            "credential_redactions": redactions,
            "status_code": getattr(exc, "status_code", None),
            "provider_kind": getattr(getattr(exc, "kind", None), "value", None),
            "scope": "native error, not full failed HTTP response bytes",
        }

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        rid = self.receipts.begin(request)
        request = replace(request, extra_headers={**(request.extra_headers or {}), "X-DataDesigner-Request-Id": rid})
        try:
            response = self._inner.completion(request)
        except BaseException as exc:
            self.receipts.uncertain(
                rid,
                type(exc).__name__,
                error=self._error_evidence(exc, request),
            )
            raise
        self.receipts.response(rid, response)
        return response

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        rid = self.receipts.begin(request)
        request = replace(request, extra_headers={**(request.extra_headers or {}), "X-DataDesigner-Request-Id": rid})
        try:
            response = await self._inner.acompletion(request)
        except BaseException as exc:
            self.receipts.uncertain(
                rid,
                type(exc).__name__,
                error=self._error_evidence(exc, request),
            )
            raise
        self.receipts.response(rid, response)
        return response

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise CallBlocked("This receipt policy supports chat only")

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise CallBlocked("This receipt policy supports chat only")

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise CallBlocked("This receipt policy supports chat only")

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise CallBlocked("This receipt policy supports chat only")


def recorded_model_registry(
    *,
    model_configs: list[ModelConfig],
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    receipts: CallReceipts,
    client_concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
    request_admission: AdaptiveRequestAdmissionController | None = None,
) -> ModelRegistry:
    """Use the existing ModelRegistry facade-factory extension point.

    Install on an engine ResourceProvider before native dataset building. This
    does not change DataDesigner's default public constructor or global retries.
    A shared admission controller keeps NeMo responsible for concurrency.
    """
    admission = request_admission or AdaptiveRequestAdmissionController()
    retry = RetryConfig(max_retries=0)

    def facade_factory(
        config: ModelConfig,
        resolver: SecretResolver,
        providers: ModelProviderRegistry,
        retry_config: RetryConfig | None,
    ) -> ModelFacade:
        provider = providers.get_provider(config.provider)
        policy = receipts.policy
        if (provider.name, provider.endpoint, config.model) != (policy.provider_name, policy.endpoint, policy.model_id):
            raise ValueError("Model route does not match the frozen call policy")
        client = create_model_client(
            config,
            resolver,
            providers,
            retry_config=retry,
            client_concurrency_mode=client_concurrency_mode,
            request_admission=admission,
        )
        credential = resolver.resolve(provider.api_key) if provider.api_key else None
        credentials = (credential,) if isinstance(credential, str) and credential else ()
        return ModelFacade(
            config, providers, client=RecordedChatClient(client, receipts, credential_values=credentials)
        )

    return ModelRegistry(
        model_configs=model_configs,
        secret_resolver=secret_resolver,
        model_provider_registry=model_provider_registry,
        model_facade_factory=facade_factory,
        request_admission=admission,
        retry_config=retry,
    )

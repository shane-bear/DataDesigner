# Recorded chat calls for evidence-rich seeds

This opt-in consumer integration continues the offline seed recipe. It uses the
existing `ModelRegistry(model_facade_factory=...)`, native HTTP adapters and
native request admission. It does not replace the NeMo scheduler, seed readers,
checkpoint files, judges, content review or training admission. No engine source
changes are needed.

## Configure, then build with native NeMo

0. Set `NEMO_TELEMETRY_ENABLED=false` **before Python imports**. NeMo otherwise
   enables vendor telemetry by default. `otel_metrics_port=None` disables only
   the local OpenTelemetry metrics endpoint, not this separate HTTP telemetry.
   Synthetic HTTP tests intercepted that extra POST; it was not sent externally.
1. Freeze authorized inputs, endpoint, model ID, price reference, input-token
   ceiling, output-token ceiling and run-dollar cap. `CallPolicy` stores those
   assumptions, **not current commercial prices or an endpoint guarantee**.
2. Open `CallReceipts(Path(...), policy)` inside the protected run-artifact
   directory. Keep it open until all native tasks finish. It is a POSIX
   single-writer file; threads share a lock. No credential files are read by it.
3. Construct `recorded_model_registry(...)` with the ordinary model configs,
   provider registry and secret resolver. For an async native dataset builder,
   explicitly pass `ClientConcurrencyMode.ASYNC`.
4. Pass that registry as the public `ResourceProvider.model_registry` field
   before constructing the native `DatasetBuilder`. The regression test shows
   a complete text-column -> native execution -> Parquet path. This is the
   engine's existing injection point, **not automatic wiring into the default
   public `DataDesigner(...)` constructor**. Do not monkeypatch global factories.
5. Keep `skip_health_check=True` on the model configs when no startup model call
   is authorized. The native default is false: two generated rows can make
   **three** calls. All calls through this registry, including health checks,
   count against the same cap. Parser correction/restart policies also require
   explicit configuration; use zero for first-answer experiments.
6. Close the registry (async `aclose()` in async mode), then close receipts.

`RecordedChatClient` accepts chat only. Merely wrapping a client with hidden
retries does **not** disable them: use the registry helper, which explicitly sets
`RetryConfig(max_retries=0)` on the native executor while native request admission
also disables adapter transport retries. HTTP 429/500, lost responses and timeouts
are not automatically resent. This does not establish exactly-once execution at
a remote provider. A request ID header is correlation, not an idempotency promise.

## Receipts and honest budget accounting

Before dispatch, append and fsync a request identity and a reservation:

```
reserved USD = (input ceiling * input price + output ceiling * output price) / 1e6
admit only if used USD + outstanding reservations + this reservation <= cap
```

The first native response (including provider JSON in `raw`, where available)
and usage are appended without stripping text or repairing content. Credential
headers are never written. Native provider errors retain class/status/kind and the native error message;
known provider credentials and extra-header values are redacted if reflected in
that message. Record redaction counts instead of dropping all error signal.
Full failed HTTP response bodies are not available at this boundary. This is not
a wire-level HTTP archive, and malformed HTTP success bodies may have no saved
native response. Never claim otherwise. Receipt admission means cost-accounted,
not that an answer was returned; `responses_saved` reports that separately.

Missing/invalid usage, cancellation, failed calls, write failures and interrupted
reservations do not silently become zero-cost calls. Unknown usage holds the
full reservation and stops new calls. Known usage above a ceiling records the
actual modeled cost and stops for reconciliation; it cannot retroactively prevent
overspend. Even normal costs are **token-price estimates, not invoices**. Token
ceilings, provider token semantics and prices must be established before a real
paid run. This synthetic milestone proves local accounting logic, not a hard
commercial billing guarantee.

Receipts use restrictive new-file permissions and a hash chain to detect
accidental edits. They contain authorized input/output bodies and must remain
private alongside the corpus. Hashes are not semantic validation. Partial or
corrupt files are not truncated or repaired automatically. Write failures stop
further dispatch; a response that could not be durably written remains unknown.

Reopening replays only local receipt events, **never POSTs**. It treats unfinished
reservations as undecidable. `summary()` reports call-receipt admitted,
excluded-with-reason and undecidable counts plus one evidence-needs list. Those
counts are not packets, tasks, correct extractions or training samples.

`reconcile([...])` accepts one batch of provider usage/billing or non-acceptance
evidence references and amounts. References are operator assertions, not machine-
verified evidence; original response events stay intact. Settlements do not prove
content quality or authorize a rerun. They must be reviewed before further paid
work. Do not ask an operator to classify individual sessions or select signal.

## Validation scope

Tests cover the actual ModelRegistry -> ModelFacade -> request executor -> native
HTTP adapter path; a loopback server logs an accepted POST before deliberately
closing the connection. Other faults use synthetic HTTP boundaries. Native
DatasetBuilder writes Parquet with the guarded registry. No real model quality,
DeepSeek protocol compatibility, official tokenizer, invoice accuracy,
independent content review or training benefit is established by these tests.

For the entire upstream test suite only, its attribution-header tests assume
telemetry is enabled. Validate those unchanged assumptions with
`NEMO_TELEMETRY_ENABLED=true` and redirect `NEMO_TELEMETRY_ENDPOINT` to a local
non-service URL, never the vendor endpoint. This integration's tests still apply
the native opt-out flags locally. Do not copy that whole-suite test environment
to a real corpus run. A first whole-suite attempt with the global opt-out failed
six upstream header tests; those original failures are retained in run evidence.

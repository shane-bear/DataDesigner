# External development history: first offline NeMo milestone

This is an **experimental consumer integration**, not a replacement DataDesigner
engine, raw-history importer, training release, or validated semantic cleaner.
The operator-specific confirmed project contract is retained in the private
research workspace, not published in this public fork. Its HTML is sealed;
implementation status is maintained in its authoritative Markdown only.

## What is implemented

- `data_designer.integrations.external_history`: typed evidence packet,
  native-object anchors, facts/optional lessons, coverage partition and binding
  result schemas.
- `packet_from_occurrences`: thin adapter over **existing reader output**. It
  anchors every JSON leaf/empty container by rule, preserves scalar types and
  native null IDs, and distinguishes equal content in different occurrences.
  It does not ask a person to choose useful records or fields.
- `offline_config`: native `LocalFileSeedSource` → local callable validation →
  expression columns → native dataset artifacts. No independent runner, queue,
  checkpoint implementation, review service, or truth database.
- `generation_config`: **declarations only**, mapping proposed extraction,
  binding validation and source-visible categorical judge to native columns.
  No generation/judge endpoint has been executed or accepted in this milestone.
- `scripts/external_history_offline.py`: low-level replay entry consuming **all
  rows of a normalized seed file**. The existing discovery/coverage ledger must
  drive this entry; asking a human to locate signal-bearing sessions is not the
  main path. Raw Claude/Codex/OpenCode readers are not duplicated here.

Each seed row has `packet_json` (an `EvidencePacket`) and `extraction_json` (the
unchanged first proposed extraction, or an explicitly synthetic test fixture).
Failed and partial rows remain in the output.

## Reproduce offline

Use the repository's frozen workspace environment:

```sh
uv sync --frozen --all-packages --group dev
uv run pytest -q packages/data-designer/tests/integrations/test_external_history.py \
  packages/data-designer/tests/integrations/test_external_history_resume.py
uv run python scripts/external_history_offline.py \
  --seed-jsonl <manifest-produced-seed.jsonl> \
  --artifact-root <research-work-directory>/offline-artifacts
```

The six-row synthetic integration fixture yields mechanical-stage counts
`admitted=3`, `excluded=2`, `undecidable=1`. These are **packet rows**, not six
independent tasks or six training samples. Record occurrences inside packets
are counted separately and include repeated views.

`admitted` here means **binding checks allow the next source-review stage**.
It is not semantic correctness, permission to transfer data, content acceptance,
or training admission. `content_review_status` remains `not_run` and
`training_admitted` remains false. The script reports one undecidable list with
what evidence would settle each item; it never silently repairs first answers.

The current DataDesigner interface requires a provider/model declaration even
without LLM columns. The offline route uses an unused loopback provider named
`offline-unused`; no key is supplied. The real engine test rejects HTTP at its
external transport boundary and still produces Parquet. A temporary local
metrics endpoint may be started by DataDesigner's default process telemetry;
it is not a model request or a persistent deployed service.

## Boundaries observed, not hidden

- First bootstrap used `set -o pipefail` under `/bin/sh` and exited before any
  mutation. The corrected background command used POSIX shell options.
- Initial offline integration rejected an empty provider list. It then exposed
  a missing explicit `validator_type`; both first failures are retained.
- The validator checks canonical native-object hashes and typed pointers,
  **not raw-file authenticity or semantic entailment**. A true quote/anchor can
  still support an incorrect interpretation.
- Native completed-row-group resume was exercised from one to six rows; the
  original Parquet bytes and ordered seed occurrences survived. This does not
  prove interrupted model POSTs are exactly-once.
- A real native retry transport with a synthetic accepted-then-lost HTTP
  response attempted POST four times with `max_retries=3`, once with zero.
  This setting's end-to-end model-client injection and first-answer receipts
  remain unfinished. Do not assume a model/provider extra-body flag controls it.
- The upstream full test gate needs `jq`; its unreadable-file case must run
  without root's DAC bypass. Non-root Ruff also needs a writable cache directory.
  These environmental failures were retained, not skipped/quarantined or fixed
  by weakening product assertions.

## What is deliberately NOT implemented or authorized yet

- No real-history reader-to-seed end-to-end import or raw-file hash replay.
- No Flash protocol/structured-generation/source-judge acceptance, independent
  content-quality evaluation, full-corpus semantic recall, or cost comparison.
- No integrated dollar reservation guard, uncertain-POST receipt reconciliation,
  or original first-answer capture across native retries.
- No full-corpus external transfer, spending cap approval, training-policy/base/
  template decision, training dataset release, model training, deployment,
  hot reload, or container lifecycle action.

Before model execution, existing broad discovery/sensitivity classification
must produce the authorized safe queue; withheld/undecidable material stays
local. Freeze source, config, prompts, model route, token/call/USD limits and
receipt policy first. The generation declaration is **not** a production-ready
execution entry. ReMe-style lesson extraction remains optional: direct QA and
tool-trajectory views need not be forced through a summary card.

The work is based on fork commit
`3febeef70cccb496cdf87d9709c8870bcff243a1`. The older 24 reader tests from
`207eda6...` are not inherited as acceptance for this branch.

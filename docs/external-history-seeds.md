# Discovery-led reader references as native NeMo seeds

This continues the offline seed and call-receipt integrations. An existing broad
preparation directory drives the bridge through `preparation-seal.json`,
`analysis-contract.json`, its declared derived-input hashes and the complete
`occurrence-evidence-index.jsonl`. The CLI takes a preparation root, **not a list
of sessions selected for useful signal**. It does not discover fresh host files
or reinterpret a previous evaluation's candidate list as the learning corpus.

```
NEMO_TELEMETRY_ENABLED=false PYTHONPATH=<feature source paths> \
  python scripts/external_history_seeds.py \
  --preparation-root <existing preparation directory> \
  --output-dir <new private run directory> --native-roundtrip
```

The output directory must be new. Failed attempts remain separate artifacts.
Only generic integration code and synthetic tests belong in a public repository;
real seeds, filenames, identities, hashes and receipts remain private.

## What is preserved and checked

- Rehash the declared **derived** inputs and preparation outputs. Never open a
  raw locator, compressed history or SQLite database. Source file hashes are
  archived inventory references, **not freshly recomputed source hashes**.
- Join every registered occurrence to its reader atom and file inventory.
  Metadata disagreement, missing joins and sensitive/unknown source decisions
  remain undecidable with evidence needs. Duplicate identities stop the export
  rather than silently merging executions. Empty native IDs and scalar types
  are not invented or converted to text.
- Each seed row is one reader occurrence: identity, source group, locator,
  native-ID metadata, order, links, timestamp, three distinct hash scopes,
  classification and a byte-offset reference to the sealed reader artifact.
  Sensitive holds are opaque metadata rows, not model-ready bodies.
- The native LocalFileSeedSource -> local ValidationColumnConfig -> Parquet
  route preserves **all** supplied seed rows, including undecidable holds.
  The CLI compares every ordered seed string after loading the Parquet output.
  Local validation checks metadata, not original meaning or historical success.
- `ReaderViewStore` resolves admitted views only from its explicit sealed reader
  artifact, never from raw locator paths. Hash-check the artifact once, then
  validate each identity and canonical cleaned-view hash at its byte offset.

## Existing reader loss is not repaired by a bridge

The legacy reader's `clean()` omits fields such as signatures/encrypted material,
replaces nontext payloads, and sometimes emits only message content rather than
an entire native event. Its payload is a **cleaned reader view**, not a full native
record. Keep these limits visible. A source-locator binding is not raw authenticity
or semantic entailment. A future evidence-rich reader revision needs its own
regressions and authorized raw-record replay; this bridge does not waive that.

A reproduced legacy hash problem is particularly important: the reader hashes
insertion-order JSON, then its writer sorts nested keys. The old payload hash may
therefore be unreplayable from the saved payload despite an unchanged whole-file
seal. Never rewrite it or call it a canonical hash. Preserve both:

1. `payload_sha256_recorded`: the opaque legacy value copied from the index;
2. `payload_view_sha256`: a new canonical hash of the saved cleaned view;
3. `legacy_hash_replay`: match / unreplayable-after-sort / not-attempted-held.

Admission here means **manifest-bound reference/metadata admission only**. It
must not hide legacy fidelity gaps, approve commercial transfer, certify complete
sessions/tasks, close sensitive holds, or produce training admission.

## Source identity, work identity and HTTP attempts

`occurrence_id` hashes provider, namespace, native-session metadata and locator,
not payload text. `input_revision` covers the complete seed metadata. A
`WorkBinding` additionally freezes stage and prompt/contract hash. Distinct
source executions with identical text remain distinct; repeated attempts for one
work item share work identity but have different request IDs. No response cache
or exactly-once remote execution is implied.

`bound_prompt(binding, evidence)` supplies a declared outer JSON envelope with
`DATA_DESIGNER_SEED_JSON\n`. The non-literal prefix is needed: native Jinja can
otherwise turn a bare JSON object into Python dict repr, losing the JSON contract.
Use a native seed field and an LLM column prompt `{{ bound_prompt }}`. Populate
`ResourceProvider.seed_reader` as well as its recorded model registry when using
the low-level native builder. The default DataDesigner constructor is not
silently monkeypatched.

The call boundary reads only that prefix and the **entire first user message**,
not quoted history or nested lookalike keys. Malformed declared envelopes stop
before dispatch. Valid declared metadata is appended beside the unique HTTP
attempt ID. It is correlation, **not verified source evidence or authorization**.
The native-column regression demonstrates seed -> rendered prompt -> real native
client at a synthetic HTTP boundary -> receipt mapping, with identical text in
two source occurrences. It does not demonstrate real model content quality.

## Output and remaining evidence

`report.json` separates source-occurrence counts, seed-row counts, join gaps,
provider/group counts, file-inventory exclusions and legacy-hash limitations.
Do not add these different units together. One `needs` list names source gaps
and evidence to settle them; operators do not select useful sessions or resolve
items one by one. File inventory exclusions are the existing reader's scope
classification, not a claim that excluded attachments contain no useful signal.

Real protocol, privacy/transfer and spending scope, full-native reconstruction,
task/attempt context closure, independent content review, training release and
real development benefit remain separate work. A local Parquet round trip does
not approve any of them.

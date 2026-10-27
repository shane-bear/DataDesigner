# SPDX-FileCopyrightText: Copyright (c) 2026 DataDesigner contributors
# SPDX-License-Identifier: Apache-2.0

"""Bridge sealed derived reader occurrences to native NeMo seed manifests.

No raw source path is opened. Metadata/reader-view integrity is not full-native
fidelity, source authenticity, semantic admission or external-transfer approval.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

import data_designer.config as dd
import data_designer.lazy_heavy_imports as lazy
from data_designer.integrations.external_history import ContractModel, canonical_json

PROVIDERS = {"claude": "claude_code", "claude_code": "claude_code", "codex": "codex", "opencode": "opencode"}
BINDING_KEY = "data_designer_work"
BINDING_PREFIX = "DATA_DESIGNER_SEED_JSON\n"
SHA_PATTERN = r"^[a-f0-9]{64}$"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def reader_payload_sha256(value: Any) -> str:
    """Legacy reader hashes insertion-order JSON, deliberately NOT canonical JSON."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class WorkBinding(ContractModel):
    schema_version: Literal[1] = 1
    occurrence_id: str = Field(pattern=r"^O-[a-f0-9]{64}$")
    input_revision: str = Field(pattern=SHA_PATTERN)
    stage: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=SHA_PATTERN)
    work_id: str = Field(pattern=r"^W-[a-f0-9]{64}$")

    @model_validator(mode="after")
    def check_work_identity(self) -> WorkBinding:
        expected = "W-" + object_sha256([self.occurrence_id, self.input_revision, self.stage, self.contract_sha256])
        if self.work_id != expected:
            raise ValueError("work_identity_mismatch")
        return self


def work_binding(occurrence_id: str, input_revision: str, stage: str, contract_sha256: str) -> WorkBinding:
    return WorkBinding(
        occurrence_id=occurrence_id,
        input_revision=input_revision,
        stage=stage,
        contract_sha256=contract_sha256,
        work_id="W-" + object_sha256([occurrence_id, input_revision, stage, contract_sha256]),
    )


def bound_prompt(binding: WorkBinding, evidence: JsonValue) -> str:
    """A declared outer envelope, not a search for markers inside quoted history."""
    # A non-literal prefix prevents native Jinja rendering from converting a
    # JSON object into Python dict repr (which would lose the JSON contract).
    return BINDING_PREFIX + canonical_json({BINDING_KEY: binding.model_dump(mode="json"), "evidence": evidence})


def binding_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Read only the first user message's entire outer JSON envelope.

    This is correlation metadata, not proof of authenticity or authorization.
    Later history/tool messages and nested lookalike keys are not searched.
    """
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            if len(content) != 1 or not isinstance(content[0], dict) or content[0].get("type") != "text":
                return None
            content = content[0].get("text")
        if not isinstance(content, str) or not content.startswith(BINDING_PREFIX):
            return None
        try:
            body = json.loads(content[len(BINDING_PREFIX) :])
        except (ValueError, TypeError) as error:
            raise ValueError("invalid_declared_work_envelope") from error
        if not isinstance(body, dict) or BINDING_KEY not in body:
            raise ValueError("missing_declared_work_binding")
        return WorkBinding.model_validate(body[BINDING_KEY]).model_dump(mode="json")
    return None


class SeedOccurrence(ContractModel):
    occurrence_id: str = Field(pattern=r"^O-[a-f0-9]{64}$")
    input_revision: str = Field(pattern=SHA_PATTERN)
    atom_id: str
    provider: str
    source_group: str
    source_namespace: str
    native_session_id: JsonValue
    native_message_id: JsonValue
    native_role: JsonValue
    reader_emission_ordinal: int = Field(ge=0)
    locator: dict[str, JsonValue]
    timestamp: JsonValue
    tool_links: list[JsonValue]
    record_kind: str
    record_type: str
    origin: str
    fixture_cohort: bool
    raw_file_sha256_recorded: str | None = Field(default=None, pattern=SHA_PATTERN)
    raw_record_sha256_recorded: str | None = Field(default=None, pattern=SHA_PATTERN)
    payload_sha256_recorded: str | None = Field(default=None, pattern=SHA_PATTERN)
    payload_view_sha256: str | None = Field(default=None, pattern=SHA_PATTERN)
    legacy_hash_replay: Literal["matches_decoded_view", "unreplayable_after_sorted_serialization", "not_attempted_held"]
    reader_artifact_sha256: str = Field(pattern=SHA_PATTERN)
    reader_artifact_line: int = Field(ge=1)
    reader_artifact_byte_offset: int = Field(ge=0)
    disposition: Literal["admitted", "excluded", "undecidable"]
    reasons: list[str]
    payload_view: Literal["cleaned_reader_payload_not_full_native_record"] = (
        "cleaned_reader_payload_not_full_native_record"
    )
    raw_file_verification: Literal["archived_inventory_reference_not_rehashed"] = (
        "archived_inventory_reference_not_rehashed"
    )
    training_admitted: Literal[False] = False

    @model_validator(mode="after")
    def check_identity(self) -> SeedOccurrence:
        expected = "O-" + object_sha256([self.provider, self.source_namespace, self.native_session_id, self.locator])
        if self.occurrence_id != expected:
            raise ValueError("source_occurrence_identity_mismatch")
        if self.input_revision != object_sha256(self.model_dump(mode="json", exclude={"input_revision"})):
            raise ValueError("seed_revision_mismatch")
        return self


@dataclass(frozen=True)
class DerivedInputs:
    index: Path
    atoms: Path
    inventory: Path
    expected_hashes: dict[str, str]
    contract_sha256: str

    @classmethod
    def from_preparation(cls, root: Path) -> DerivedInputs:
        """Existing discovery/contract drives all items; no session/file signal selection."""
        seal = json.loads((root / "preparation-seal.json").read_text())
        for name, expected in seal["outputs"].items():
            if Path(name).name != name or file_sha256(root / name) != expected:
                raise ValueError("preparation_seal_mismatch")
        contract = json.loads((root / "analysis-contract.json").read_text())
        declared = contract["derived_input_hashes"]

        def unique(name: str) -> Path:
            matches = [Path(p) for p in declared if Path(p).name == name]
            if len(matches) != 1:
                raise ValueError("ambiguous_or_missing_derived_artifact:" + name)
            return matches[0]

        atoms, inventory = unique("atoms.jsonl"), unique("input-inventory.jsonl")
        # Rehash declared derived artifacts, never contract.snapshot/raw paths.
        checked = {}
        for path, expected in declared.items():
            if file_sha256(Path(path)) != expected:
                raise ValueError("derived_input_hash_mismatch")
            checked[path] = expected
        checked[str(root / "occurrence-evidence-index.jsonl")] = seal["outputs"]["occurrence-evidence-index.jsonl"]
        return cls(root / "occurrence-evidence-index.jsonl", atoms, inventory, checked, seal["contract_sha256"])


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def export_seeds(inputs: DerivedInputs, output: Path) -> dict[str, Any]:
    """Join ALL indexed occurrences and retain every decision, with opaque holds.

    Seed rows contain references/metadata, not payload bodies. All admitted
    reader views are hash-checked here; no field is picked for semantic signal.
    Missing/extra/duplicate joins are surfaced rather than silently discarded.
    """
    for path, expected in inputs.expected_hashes.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError("input_changed_before_export")
    output.mkdir(parents=True, exist_ok=False)
    index = {}
    for record in _rows(inputs.index):
        if record["atom_id"] in index:
            raise ValueError("duplicate_index_atom_identity")
        index[record["atom_id"]] = record
    inventory = {}
    for record in _rows(inputs.inventory):
        if record["path"] in inventory:
            raise ValueError("duplicate_inventory_path")
        inventory[record["path"]] = record
    counts: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    reasons_count: Counter[str] = Counter()
    legacy_hash_counts: Counter[str] = Counter()
    seen_atoms, seen_occurrences, groups = set(), set(), set()
    needs = []
    atoms_hash = inputs.expected_hashes[str(inputs.atoms)]
    with inputs.atoms.open("rb") as atoms, (output / "seeds.jsonl").open("x") as seeds:
        ordinal = 0
        while True:
            offset = atoms.tell()
            line = atoms.readline()
            if not line:
                break
            ordinal += 1
            atom = json.loads(line)
            aid = atom["atom_id"]
            if aid in seen_atoms:
                raise ValueError("duplicate_reader_atom_identity")
            seen_atoms.add(aid)
            meta = index.get(aid)
            if meta is None:
                needs.append(
                    {
                        "atom_id": aid,
                        "reason": "reader_atom_missing_from_index",
                        "evidence_needed": "Reconciled discovery index including this reader occurrence",
                    }
                )
                continue
            provider = PROVIDERS.get(meta["provider"], meta["provider"])
            reasons = []
            held = meta["learning_analysis_disposition"] != "admitted_analysis"
            if held:
                reasons.append("sensitive_or_unknown_discovery_hold")
            for key in (
                "provider",
                "session_key",
                "source_namespace",
                "source_session_id",
                "native_message_id",
                "native_role",
                "locator",
                "payload_sha256",
                "raw_record_sha256",
                "tool_links",
                "record_kind",
                "record_type",
                "timestamp",
            ):
                if atom.get(key) != meta.get(key):
                    reasons.append("reader_index_mismatch:" + key)
            if provider not in set(PROVIDERS.values()):
                reasons.append("unknown_provider")
            raw_file = inventory.get(meta["locator"]["path"])
            if raw_file is None:
                reasons.append("raw_file_missing_from_archived_inventory")
            view_hash = None
            legacy_replay = "not_attempted_held"
            if not held:
                if atom["payload"] is None or atom.get("status") != "admitted_observation":
                    reasons.append("reader_payload_unavailable")
                else:
                    # Existing writer sorted nested keys after its insertion-order
                    # hash. Preserve that opaque recorded value, never rewrite it.
                    view_hash = object_sha256(atom["payload"])
                    legacy_replay = (
                        "matches_decoded_view"
                        if reader_payload_sha256(atom["payload"]) == meta["payload_sha256"]
                        else "unreplayable_after_sorted_serialization"
                    )
            legacy_hash_counts[legacy_replay] += 1
            identity = "O-" + object_sha256(
                [provider, meta["source_namespace"], meta["source_session_id"], meta["locator"]]
            )
            if identity in seen_occurrences:
                raise ValueError("ambiguous_source_occurrence_identity")
            seen_occurrences.add(identity)
            seed = SeedOccurrence.model_construct(
                occurrence_id=identity,
                input_revision="0" * 64,
                atom_id=aid,
                provider=provider,
                source_group=meta["session_key"],
                source_namespace=meta["source_namespace"],
                native_session_id=meta["source_session_id"],
                native_message_id=meta["native_message_id"],
                native_role=meta["native_role"],
                reader_emission_ordinal=meta["reader_emission_ordinal"],
                locator=meta["locator"],
                timestamp=meta["timestamp"],
                tool_links=meta["tool_links"],
                record_kind=meta["record_kind"],
                record_type=meta["record_type"],
                origin=meta["origin"],
                fixture_cohort=meta["fixture_cohort"],
                raw_file_sha256_recorded=raw_file["sha256"] if raw_file else None,
                raw_record_sha256_recorded=meta["raw_record_sha256"],
                payload_sha256_recorded=meta["payload_sha256"],
                payload_view_sha256=view_hash,
                legacy_hash_replay=legacy_replay,
                reader_artifact_sha256=atoms_hash,
                reader_artifact_line=ordinal,
                reader_artifact_byte_offset=offset,
                disposition="undecidable" if reasons else "admitted",
                reasons=reasons or ["derived_view_manifest_bound_not_content_reviewed"],
            )
            data = seed.model_dump(mode="json")
            data["input_revision"] = object_sha256({k: v for k, v in data.items() if k != "input_revision"})
            seed = SeedOccurrence.model_validate(data)
            counts[seed.disposition] += 1
            providers[provider] += 1
            groups.add(seed.source_group)
            reasons_count.update(reasons)
            if reasons:
                needs.append(
                    {
                        "occurrence_id": identity,
                        "atom_id": aid,
                        "reasons": reasons,
                        "evidence_needed": "Reconciled reader/index/inventory or approved sensitivity evidence; never resolve by guessing or opening raw files during a pause",
                    }
                )
            seeds.write(canonical_json({"seed_json": seed.model_dump_json()}) + "\n")
    for aid in index.keys() - seen_atoms:
        needs.append(
            {
                "atom_id": aid,
                "reason": "indexed_occurrence_missing_from_reader",
                "evidence_needed": "The matching sealed reader occurrence or an explicit documented source gap",
            }
        )
    report = {
        "unit": "reader_occurrences_not_tasks_or_training_examples",
        "registered_index_occurrences": len(index),
        "reader_occurrences": len(seen_atoms),
        "seed_rows": sum(counts.values()),
        "admitted": counts["admitted"],
        "excluded": counts["excluded"],
        "excluded_with_reason": {},
        "undecidable": len(needs),
        "seed_undecidable": counts["undecidable"],
        "registered_source_union": len(set(index) | seen_atoms),
        "join_gap_entries": len(needs) - counts["undecidable"],
        "providers": dict(providers),
        "source_groups": len(groups),
        "legacy_hash_replay_counts": dict(legacy_hash_counts),
        "legacy_hash_fidelity_need": "Original pre-sorted serialization or authorized raw-record replay is needed to reproduce old hashes; derived file seal binds only the current cleaned view",
        "reason_counts": dict(reasons_count),
        "needs": needs,
        "input_hashes": inputs.expected_hashes,
        "file_inventory_counts": dict(Counter(item["status"] for item in inventory.values())),
        "file_exclusion_reasons": dict(Counter(item["reason"] for item in inventory.values() if "reason" in item)),
        "seed_sha256": file_sha256(output / "seeds.jsonl"),
        "payload_fidelity": "cleaned reader views only; native fields may already be omitted",
        "raw_sources_opened": False,
        "training_admitted": 0,
        "model_calls": 0,
    }
    for path, expected in inputs.expected_hashes.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError("input_changed_during_export")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def validate_seed_manifest(frame: Any) -> Any:
    """Native local validation of seed metadata; NEVER dereference raw locators."""
    results = []
    for value in frame["seed_json"]:
        try:
            row = SeedOccurrence.model_validate_json(value)
            results.append(
                {"is_valid": row.disposition == "admitted", "disposition": row.disposition, "reasons": row.reasons}
            )
        except ValueError:
            results.append({"is_valid": False, "disposition": "undecidable", "reasons": ["invalid_seed_metadata"]})
    return lazy.pd.DataFrame(results, index=frame.index)


def seed_manifest_config(path: Path) -> dd.DataDesignerConfigBuilder:
    config = dd.DataDesignerConfigBuilder()
    config.with_seed_dataset(dd.LocalFileSeedSource(path=str(path.resolve())))
    config.add_column(
        dd.ValidationColumnConfig(
            name="seed_validation",
            target_columns=["seed_json"],
            validator_type=dd.ValidatorType.LOCAL_CALLABLE,
            validator_params=dd.LocalCallableValidatorParams(validation_function=validate_seed_manifest),
        )
    )
    return config


class ReaderViewStore:
    """Resolve admitted references from ONE sealed derived reader artifact only.

    Raw locator paths are never opened. Returned payloads remain cleaned views,
    not original native objects. Use a separate instance per worker/thread.
    """

    def __init__(self, path: Path, expected_sha256: str) -> None:
        if file_sha256(path) != expected_sha256:
            raise ValueError("reader_artifact_hash_mismatch")
        self.sha256 = expected_sha256
        self._stream = path.open("rb")

    def close(self) -> None:
        self._stream.close()

    def resolve(self, seed: SeedOccurrence) -> JsonValue:
        if seed.disposition != "admitted":
            raise ValueError("held_or_unresolved_reader_view")
        if seed.reader_artifact_sha256 != self.sha256:
            raise ValueError("reader_reference_artifact_mismatch")
        self._stream.seek(seed.reader_artifact_byte_offset)
        atom = json.loads(self._stream.readline())
        if atom["atom_id"] != seed.atom_id or object_sha256(atom["payload"]) != seed.payload_view_sha256:
            raise ValueError("reader_reference_content_mismatch")
        return atom["payload"]

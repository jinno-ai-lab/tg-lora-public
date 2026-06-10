from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from torch.utils.data import Sampler

from src.utils.io import load_jsonl


def _canonical_record_bytes(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_record(record: dict[str, Any]) -> str:
    return hashlib.sha1(_canonical_record_bytes(record)).hexdigest()


def hash_records(records: Sequence[dict[str, Any]]) -> str:
    payload = b"\n".join(hash_record(record).encode("ascii") for record in records)
    return hashlib.sha1(payload).hexdigest()


def batch_key_from_sample_keys(sample_keys: Sequence[str]) -> str:
    payload = ",".join(sample_keys).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def build_epoch_batches(sample_count: int, batch_size: int) -> list[list[int]]:
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batches: list[list[int]] = []
    for start in range(0, sample_count, batch_size):
        batches.append(list(range(start, min(start + batch_size, sample_count))))
    return batches


def _extract_record_id(record: dict[str, Any]) -> str | None:
    for key in ("id", "record_id", "source_id", "uid", "uuid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def _extract_record_text(record: dict[str, Any]) -> str:
    text = str(record.get("text", "") or "")
    if text:
        return text
    prompt = str(record.get("prompt", "") or "")
    completion = str(record.get("completion", "") or "")
    return prompt + completion


@dataclass(frozen=True)
class SampleLocator:
    sample_key: str
    dataset_index: int
    record_id: str | None = None
    text_sha1: str | None = None
    text_preview: str | None = None

    @classmethod
    def from_record(
        cls,
        *,
        sample_key: str,
        dataset_index: int,
        record: dict[str, Any] | None,
    ) -> "SampleLocator":
        if record is None:
            return cls(sample_key=sample_key, dataset_index=dataset_index)
        text = _extract_record_text(record)
        return cls(
            sample_key=sample_key,
            dataset_index=dataset_index,
            record_id=_extract_record_id(record),
            text_sha1=(
                hashlib.sha1(text.encode("utf-8")).hexdigest() if text else None
            ),
            text_preview=(text[:120] if text else None),
        )


@dataclass(frozen=True)
class BatchLocator:
    batch_key: str
    batch_index: int
    dataset_indices: list[int]
    sample_keys: list[str]


@dataclass(frozen=True)
class DeterministicBatchPlanManifest:
    format_version: int
    strategy: str
    dataset_key: str
    dataset_path: str | None
    sample_count: int
    batch_size: int
    sample_keys: list[str]
    sample_locators: list[SampleLocator]
    epoch_batches: list[list[int]]
    epoch_batch_keys: list[str]
    epoch_batch_plan_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeterministicBatchPlanManifest":
        return cls(
            format_version=int(data["format_version"]),
            strategy=str(data["strategy"]),
            dataset_key=str(data["dataset_key"]),
            dataset_path=data.get("dataset_path"),
            sample_count=int(data["sample_count"]),
            batch_size=int(data["batch_size"]),
            sample_keys=list(data["sample_keys"]),
            sample_locators=[
                SampleLocator(**locator) for locator in data.get("sample_locators", [])
            ],
            epoch_batches=[list(batch) for batch in data["epoch_batches"]],
            epoch_batch_keys=list(data["epoch_batch_keys"]),
            epoch_batch_plan_key=str(data["epoch_batch_plan_key"]),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def sample_locator_by_key(self, sample_key: str) -> SampleLocator | None:
        for locator in self.sample_locators:
            if locator.sample_key == sample_key:
                return locator
        return None

    def batch_locator_by_key(self, batch_key: str) -> BatchLocator | None:
        for batch_index, current_key in enumerate(self.epoch_batch_keys):
            if current_key != batch_key:
                continue
            dataset_indices = list(self.epoch_batches[batch_index])
            sample_keys = [self.sample_keys[index] for index in dataset_indices]
            return BatchLocator(
                batch_key=batch_key,
                batch_index=batch_index,
                dataset_indices=dataset_indices,
                sample_keys=sample_keys,
            )
        return None

    def batch_locator_at_position(self, batch_position: int) -> BatchLocator:
        if batch_position < 0:
            raise ValueError(
                f"batch_position must be non-negative, got {batch_position}"
            )
        if not self.epoch_batch_keys:
            raise ValueError("manifest has no epoch batches")
        epoch_batch_index = batch_position % len(self.epoch_batch_keys)
        batch_key = self.epoch_batch_keys[epoch_batch_index]
        locator = self.batch_locator_by_key(batch_key)
        if locator is None:
            raise ValueError(f"batch key missing from manifest: {batch_key}")
        return locator


def build_deterministic_batch_plan_manifest(
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    strategy: str = "dataset_order_repeat",
    dataset_path: str | None = None,
) -> DeterministicBatchPlanManifest:
    if not records:
        raise ValueError("records must not be empty")
    if strategy != "dataset_order_repeat":
        raise ValueError(f"Unsupported deterministic batch strategy: {strategy}")

    sample_keys = [hash_record(record) for record in records]
    epoch_batches = build_epoch_batches(len(records), batch_size)
    epoch_batch_keys = [
        batch_key_from_sample_keys([sample_keys[index] for index in batch])
        for batch in epoch_batches
    ]
    epoch_batch_plan_key = hashlib.sha1(
        "\n".join(epoch_batch_keys).encode("utf-8")
    ).hexdigest()
    return DeterministicBatchPlanManifest(
        format_version=1,
        strategy=strategy,
        dataset_key=hash_records(records),
        dataset_path=str(Path(dataset_path).resolve()) if dataset_path else None,
        sample_count=len(records),
        batch_size=batch_size,
        sample_keys=sample_keys,
        sample_locators=[
            SampleLocator.from_record(
                sample_key=sample_key,
                dataset_index=index,
                record=record,
            )
            for index, (sample_key, record) in enumerate(zip(sample_keys, records, strict=True))
        ],
        epoch_batches=epoch_batches,
        epoch_batch_keys=epoch_batch_keys,
        epoch_batch_plan_key=epoch_batch_plan_key,
    )


def build_deterministic_batch_plan_for_dataset(
    dataset: Any,
    *,
    batch_size: int,
    strategy: str = "dataset_order_repeat",
) -> DeterministicBatchPlanManifest:
    source_path = getattr(dataset, "source_path", None)
    records = getattr(dataset, "records", None)
    if isinstance(records, Sequence) and records:
        return build_deterministic_batch_plan_manifest(
            records,
            batch_size=batch_size,
            strategy=strategy,
            dataset_path=source_path,
        )

    sample_count = len(dataset)
    if sample_count <= 0:
        raise ValueError("dataset must not be empty")
    sample_keys = [
        hashlib.sha1(f"{type(dataset).__name__}:{index}".encode("utf-8")).hexdigest()
        for index in range(sample_count)
    ]
    epoch_batches = build_epoch_batches(sample_count, batch_size)
    epoch_batch_keys = [
        batch_key_from_sample_keys([sample_keys[index] for index in batch])
        for batch in epoch_batches
    ]
    epoch_batch_plan_key = hashlib.sha1(
        "\n".join(epoch_batch_keys).encode("utf-8")
    ).hexdigest()
    dataset_key = hashlib.sha1(
        f"{type(dataset).__name__}:{sample_count}".encode("utf-8")
    ).hexdigest()
    return DeterministicBatchPlanManifest(
        format_version=1,
        strategy=strategy,
        dataset_key=dataset_key,
        dataset_path=str(Path(source_path).resolve()) if source_path else None,
        sample_count=sample_count,
        batch_size=batch_size,
        sample_keys=sample_keys,
        sample_locators=[
            SampleLocator.from_record(
                sample_key=sample_key,
                dataset_index=index,
                record=None,
            )
            for index, sample_key in enumerate(sample_keys)
        ],
        epoch_batches=epoch_batches,
        epoch_batch_keys=epoch_batch_keys,
        epoch_batch_plan_key=epoch_batch_plan_key,
    )


def load_deterministic_batch_plan_manifest(
    path: str | Path,
) -> DeterministicBatchPlanManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DeterministicBatchPlanManifest.from_dict(data)


def resolve_record_for_sample_key(
    manifest: DeterministicBatchPlanManifest,
    sample_key: str,
) -> dict[str, Any] | None:
    locator = manifest.sample_locator_by_key(sample_key)
    if locator is None or manifest.dataset_path is None:
        return None

    dataset_path = Path(manifest.dataset_path)
    if not dataset_path.exists():
        return None

    records = load_jsonl(dataset_path)
    if locator.dataset_index >= len(records):
        return None
    record = records[locator.dataset_index]
    if hash_record(record) != sample_key:
        raise ValueError(
            "dataset_path no longer matches manifest sample_key at index "
            f"{locator.dataset_index}"
        )
    return record


def resolve_records_for_batch_key(
    manifest: DeterministicBatchPlanManifest,
    batch_key: str,
) -> list[dict[str, Any]] | None:
    locator = manifest.batch_locator_by_key(batch_key)
    if locator is None:
        return None
    records: list[dict[str, Any]] = []
    for sample_key in locator.sample_keys:
        record = resolve_record_for_sample_key(manifest, sample_key)
        if record is None:
            return None
        records.append(record)
    return records


def build_trajectory_key(
    *,
    mode: str,
    epoch_batch_plan_key: str,
    trainable_lora_scope: str,
    optimizer_lifecycle: str | None,
    model_name: str,
    max_seq_len: int,
    deterministic_data_order: bool,
) -> str:
    payload = {
        "mode": mode,
        "epoch_batch_plan_key": epoch_batch_plan_key,
        "trainable_lora_scope": trainable_lora_scope,
        "optimizer_lifecycle": optimizer_lifecycle,
        "model_name": model_name,
        "max_seq_len": max_seq_len,
        "deterministic_data_order": deterministic_data_order,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class DeterministicBatchSampler(Sampler[list[int]]):
    def __init__(self, epoch_batches: Sequence[Sequence[int]]) -> None:
        self._batches = [list(batch) for batch in epoch_batches]
        if not self._batches:
            raise ValueError("epoch_batches must not be empty")

    def __iter__(self) -> Iterator[list[int]]:
        yield from (list(batch) for batch in self._batches)

    def __len__(self) -> int:
        return len(self._batches)
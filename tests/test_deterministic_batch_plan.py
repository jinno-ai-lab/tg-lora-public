import json

from src.training.deterministic_batch_plan import (
    DeterministicBatchSampler, batch_key_from_sample_keys,
    build_deterministic_batch_plan_for_dataset,
    build_deterministic_batch_plan_manifest, build_epoch_batches,
    build_trajectory_key, hash_record, hash_records,
    load_deterministic_batch_plan_manifest, resolve_record_for_sample_key,
    resolve_records_for_batch_key)


def test_hash_record_is_stable_for_key_order_changes():
    left = {"text": "abc", "meta": {"x": 1, "y": 2}}
    right = {"meta": {"y": 2, "x": 1}, "text": "abc"}
    assert hash_record(left) == hash_record(right)


def test_hash_records_depends_on_record_order():
    records_a = [{"text": "a"}, {"text": "b"}]
    records_b = [{"text": "b"}, {"text": "a"}]
    assert hash_records(records_a) != hash_records(records_b)


def test_build_epoch_batches_sequential_shape():
    assert build_epoch_batches(5, 2) == [[0, 1], [2, 3], [4]]


def test_build_manifest_contains_expected_keys(tmp_path):
    records = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = build_deterministic_batch_plan_manifest(
        records,
        batch_size=2,
        dataset_path=dataset_path,
    )

    assert manifest.strategy == "dataset_order_repeat"
    assert manifest.sample_count == 3
    assert manifest.dataset_path == str(dataset_path.resolve())
    assert manifest.epoch_batches == [[0, 1], [2]]
    assert len(manifest.sample_keys) == 3
    assert manifest.sample_locators[0].dataset_index == 0
    assert manifest.sample_locators[0].text_preview == "a"
    assert manifest.epoch_batch_keys == [
        batch_key_from_sample_keys(manifest.sample_keys[:2]),
        batch_key_from_sample_keys(manifest.sample_keys[2:]),
    ]

    path = tmp_path / "batch_plan_manifest.json"
    manifest.save(path)
    loaded = load_deterministic_batch_plan_manifest(path)
    assert loaded.epoch_batch_plan_key == manifest.epoch_batch_plan_key
    assert loaded.sample_locators[0].text_preview == "a"


def test_deterministic_batch_sampler_yields_epoch_batches():
    sampler = DeterministicBatchSampler([[0, 1], [2]])
    assert list(iter(sampler)) == [[0, 1], [2]]
    assert len(sampler) == 2


def test_trajectory_key_changes_when_scope_changes():
    common = {
        "epoch_batch_plan_key": "plan",
        "optimizer_lifecycle": "recreate_per_cycle",
        "model_name": "Qwen/Qwen3.5-9B",
        "max_seq_len": 1024,
        "deterministic_data_order": True,
    }
    key_a = build_trajectory_key(
        mode="baseline",
        trainable_lora_scope="last_25_percent",
        **common,
    )
    key_b = build_trajectory_key(
        mode="baseline",
        trainable_lora_scope="all",
        **common,
    )
    assert key_a != key_b


def test_build_manifest_falls_back_to_index_keys_for_generic_dataset():
    class _Dataset:
        def __len__(self):
            return 3

    manifest = build_deterministic_batch_plan_for_dataset(_Dataset(), batch_size=2)
    assert manifest.sample_count == 3
    assert manifest.dataset_path is None
    assert manifest.epoch_batches == [[0, 1], [2]]


def test_batch_locator_at_position_wraps_repeated_epoch():
    records = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    manifest = build_deterministic_batch_plan_manifest(records, batch_size=2)
    first = manifest.batch_locator_at_position(0)
    third = manifest.batch_locator_at_position(2)
    assert first.batch_key == third.batch_key
    assert third.dataset_indices == [0, 1]


def test_sample_key_resolves_back_to_original_record(tmp_path):
    records = [
        {"id": "r1", "text": "alpha"},
        {"id": "r2", "prompt": "be", "completion": "ta"},
    ]
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = build_deterministic_batch_plan_manifest(
        records,
        batch_size=1,
        dataset_path=dataset_path,
    )

    resolved = resolve_record_for_sample_key(manifest, manifest.sample_keys[1])
    assert resolved == records[1]
    assert manifest.sample_locator_by_key(manifest.sample_keys[1]).record_id == "r2"


def test_batch_key_resolves_back_to_original_records(tmp_path):
    records = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = build_deterministic_batch_plan_manifest(
        records,
        batch_size=2,
        dataset_path=dataset_path,
    )

    batch_key = manifest.epoch_batch_keys[0]
    batch_locator = manifest.batch_locator_by_key(batch_key)
    assert batch_locator is not None
    assert batch_locator.dataset_indices == [0, 1]
    assert resolve_records_for_batch_key(manifest, batch_key) == records[:2]
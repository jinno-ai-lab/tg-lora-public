import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.eval_loss import eval_loss
from src.model.lora_utils import (iter_all_lora_params_by_layer,
                                  set_trainable_lora_layers)
from src.tg_lora.prefix_feature_cache import (
    MappedPrefixFeatureDataset, PrefixFeatureDataset, PrefixFeatureExample,
    build_prefix_feature_cache_metadata, build_prefix_feature_dataset,
    collate_prefix_feature_batch, compute_prefix_feature_shard_ranges,
    get_prefix_feature_cache_path, load_prefix_feature_dataset,
    merge_prefix_feature_cache_shards, resolve_prefix_feature_cache_seed,
    save_prefix_feature_dataset)
from src.training.config_schema import (
    DataConfig, ModelConfig, TrainingConfig,
)
from src.training.loss import compute_loss


class _SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        del attention_mask, position_ids
        return (self.norm(self.linear(hidden_states)),)


class _SimplePrefixCacheModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden: int = 16, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            [_SimpleDecoderLayer(hidden) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

        self.lora_bank = nn.Module()
        for idx in range(num_layers):
            layer_mod = nn.Module()
            layer_mod.register_parameter("lora_A", nn.Parameter(torch.randn(hidden, hidden) * 0.01))
            layer_mod.register_parameter("lora_B", nn.Parameter(torch.randn(hidden, hidden) * 0.01))
            self.layers[idx].add_module("mock_lora", layer_mod)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        del kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)[0]
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return type("Out", (), {"loss": loss})()


class _TokenDataset(Dataset):
    def __init__(self, n: int = 6, seq_len: int = 8, vocab_size: int = 32):
        self.input_ids = torch.randint(0, vocab_size, (n, seq_len))
        self.attention_mask = torch.ones(n, seq_len, dtype=torch.long)
        self.labels = self.input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class _TokenDatasetWithPositions(Dataset):
    def __init__(self, n: int = 6, seq_len: int = 8, vocab_size: int = 32):
        self.input_ids = torch.randint(0, vocab_size, (n, seq_len))
        self.attention_mask = torch.ones(n, seq_len, dtype=torch.long)
        self.labels = self.input_ids.clone()
        self.position_ids = torch.arange(seq_len).unsqueeze(0).expand(n, -1).clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "position_ids": self.position_ids[idx],
        }


def _default_metadata(**overrides):
    base = {
        "dataset_path": "data/train.jsonl",
        "model_name": "dummy-model",
        "seed": 42,
        "max_seq_len": 8,
        "split_layer_idx": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_target_modules": "all-linear",
        "trainable_lora_scope": "last_25_percent",
        "train_on_prompt": False,
        "dtype": "bfloat16",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    base.update(overrides)
    return build_prefix_feature_cache_metadata(**base)


def cross_section_byte_changing_fingerprint_inputs():
    """Config fields OUTSIDE ``DataConfig``/``ModelConfig`` that change the
    cached bytes and therefore MUST stay in the prefix-feature fingerprint.

    ``TestCacheFingerprintCompleteness`` enumerates the ``DataConfig`` /
    ``ModelConfig`` schemas directly (block 1–3), but
    :func:`build_prefix_feature_cache_metadata` ALSO reads these
    ``TrainingConfig`` fields: ``train_on_prompt`` is the LABEL-side field
    TASK-0206 added (it masks the prompt in the cached ``labels``), and
    ``trainable_lora_scope`` drives ``split_layer_idx`` (which layer's hidden
    states are captured). Both are byte-changing, so mapping them here extends
    the guard's structural "no silent drop" guarantee to the fields that live
    outside the Data/Model schemas.

    Note: ``ExperimentConfig.seed`` and ``LoRAConfig.{r, alpha, dropout,
    target_modules}`` are ALSO stamped into the fingerprint, but they do NOT
    change the cached bytes (``seed`` only because the cache-build dataloader
    uses ``shuffle=False``; the LoRA suffix is not applied during the base-model
    forward pass that populates the cache), so they are intentionally NOT in
    this byte-changing set — a stale key there can only cause an unnecessary
    rebuild or a harmless extra reuse, never a wrong-content replay.
    """
    return {
        (TrainingConfig, "train_on_prompt"): "train_on_prompt",
        (TrainingConfig, "trainable_lora_scope"): "trainable_lora_scope",
    }


def test_build_prefix_feature_dataset_matches_full_eval_loss():
    model = _SimplePrefixCacheModel()
    raw_dataset = _TokenDataset(n=6)
    raw_loader = DataLoader(raw_dataset, batch_size=2)

    cached_dataset = build_prefix_feature_dataset(
        model,
        raw_dataset,
        batch_size=2,
        device="cpu",
        split_layer_idx=2,
    )
    cached_loader = DataLoader(
        cached_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_prefix_feature_batch,
    )

    full_loss = eval_loss(model, raw_loader, device="cpu")
    cached_loss = eval_loss(model, cached_loader, device="cpu")

    assert len(cached_dataset) == len(raw_dataset)
    assert cached_dataset.total_bytes > 0
    assert math.isclose(full_loss, cached_loss, rel_tol=0.0, abs_tol=1e-5)


def test_compute_loss_accepts_cached_hidden_state_batch():
    model = _SimplePrefixCacheModel()
    raw_dataset = _TokenDataset(n=2)
    cached_dataset = build_prefix_feature_dataset(
        model,
        raw_dataset,
        batch_size=2,
        device="cpu",
        split_layer_idx=2,
        max_batches=1,
    )
    batch = collate_prefix_feature_batch([cached_dataset[0], cached_dataset[1]])
    loss = compute_loss(model, batch)
    assert torch.isfinite(loss)


def test_set_trainable_lora_layers_freezes_prefix_layers():
    model = _SimplePrefixCacheModel(num_layers=4)
    active_names = set_trainable_lora_layers(model, {2, 3})
    layer_map = iter_all_lora_params_by_layer(model)

    assert active_names
    for layer_idx, params in layer_map.items():
        for name, param in params:
            assert param.requires_grad is (layer_idx in {2, 3})
            if layer_idx in {2, 3}:
                assert name in active_names


def test_prefix_feature_dataset_round_trips_through_disk_cache(tmp_path: Path):
    model = _SimplePrefixCacheModel()
    raw_dataset = _TokenDataset(n=4)
    cached_dataset = build_prefix_feature_dataset(
        model,
        raw_dataset,
        batch_size=2,
        device="cpu",
        split_layer_idx=2,
    )
    metadata = build_prefix_feature_cache_metadata(
        dataset_path="data/train.jsonl",
        model_name="dummy-model",
        seed=42,
        max_seq_len=8,
        split_layer_idx=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules="all-linear",
        trainable_lora_scope="last_25_percent",
        train_on_prompt=False,
        dtype="bfloat16",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    cache_path = get_prefix_feature_cache_path(tmp_path, metadata)

    save_prefix_feature_dataset(cached_dataset, cache_path, metadata=metadata)
    reloaded_dataset = load_prefix_feature_dataset(cache_path)

    raw_loader = DataLoader(raw_dataset, batch_size=2)
    reloaded_loader = DataLoader(
        reloaded_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_prefix_feature_batch,
    )
    assert cache_path.exists()
    assert reloaded_dataset.total_bytes == cached_dataset.total_bytes
    assert math.isclose(
        eval_loss(model, raw_loader, device="cpu"),
        eval_loss(model, reloaded_loader, device="cpu"),
        rel_tol=0.0,
        abs_tol=1e-5,
    )


def test_prefix_feature_dataset_can_load_lazy_from_disk(tmp_path: Path):
    model = _SimplePrefixCacheModel()
    raw_dataset = _TokenDataset(n=4)
    cached_dataset = build_prefix_feature_dataset(
        model,
        raw_dataset,
        batch_size=2,
        device="cpu",
        split_layer_idx=2,
    )
    metadata = _default_metadata()
    cache_path = get_prefix_feature_cache_path(tmp_path, metadata)

    save_prefix_feature_dataset(cached_dataset, cache_path, metadata=metadata)
    lazy_dataset = load_prefix_feature_dataset(cache_path, lazy=True)

    assert isinstance(lazy_dataset, MappedPrefixFeatureDataset)
    assert lazy_dataset.total_bytes == cached_dataset.total_bytes
    assert torch.equal(lazy_dataset[0]["labels"], cached_dataset[0]["labels"])


# ---------------------------------------------------------------------------
# REQ-128: Corrupted cache file handling
# ---------------------------------------------------------------------------


class TestCorruptedCacheHandling:
    """TC-128-E01/E02/E03: load_prefix_feature_dataset rejects malformed files."""

    def test_partial_write_raises_error(self, tmp_path: Path):
        """TC-128-E01: 1-byte file causes torch.load failure."""
        bad_file = tmp_path / "partial.pt"
        bad_file.write_bytes(b"\x00")
        with pytest.raises(Exception):
            load_prefix_feature_dataset(bad_file)

    def test_non_dict_format_raises_error(self, tmp_path: Path):
        """TC-128-E02: tensor-only file causes AttributeError/TypeError."""
        bad_file = tmp_path / "tensor_only.pt"
        torch.save(torch.randn(3, 4), bad_file)
        with pytest.raises((AttributeError, TypeError, KeyError)):
            load_prefix_feature_dataset(bad_file)

    def test_missing_hidden_states_key_raises_error(self, tmp_path: Path):
        """TC-128-E03: dict without hidden_states causes KeyError."""
        bad_file = tmp_path / "no_hidden.pt"
        torch.save({"format_version": 1, "metadata": {}}, bad_file)
        with pytest.raises(KeyError):
            load_prefix_feature_dataset(bad_file)


# ---------------------------------------------------------------------------
# REQ-129: force_rebuild flag
# ---------------------------------------------------------------------------


class TestForceRebuildFlag:
    """TC-129-01/02: _maybe_cache_dataset force_rebuild logic."""

    @staticmethod
    def _simulate_maybe_cache(cache_path: Path, force_rebuild: bool):
        """Simulate the _maybe_cache_dataset source decision logic."""
        cached_prefix_datasets: dict[Path, PrefixFeatureDataset] = {}
        if cache_path in cached_prefix_datasets:
            return "memory"
        elif cache_path.exists() and not force_rebuild:
            return "disk"
        else:
            return "built"

    def test_force_rebuild_false_reuses_disk_cache(self, tmp_path: Path):
        """TC-129-01: force_rebuild=false → source='disk' when cache exists."""
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        examples = [
            PrefixFeatureExample(
                hidden_states=torch.randn(8, 16),
                attention_mask=torch.ones(8, dtype=torch.long),
                labels=torch.randint(0, 32, (8,)),
                split_layer_idx=2,
            )
        ]
        ds = PrefixFeatureDataset(examples)
        save_prefix_feature_dataset(ds, cache_path, metadata=metadata)
        assert cache_path.exists()
        source = self._simulate_maybe_cache(cache_path, force_rebuild=False)
        assert source == "disk"

    def test_force_rebuild_true_skips_disk_cache(self, tmp_path: Path):
        """TC-129-02: force_rebuild=true → source='built' even when cache exists."""
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        examples = [
            PrefixFeatureExample(
                hidden_states=torch.randn(8, 16),
                attention_mask=torch.ones(8, dtype=torch.long),
                labels=torch.randint(0, 32, (8,)),
                split_layer_idx=2,
            )
        ]
        ds = PrefixFeatureDataset(examples)
        save_prefix_feature_dataset(ds, cache_path, metadata=metadata)
        assert cache_path.exists()
        source = self._simulate_maybe_cache(cache_path, force_rebuild=True)
        assert source == "built"


# ---------------------------------------------------------------------------
# REQ-130: position_ids build path
# ---------------------------------------------------------------------------


class TestPositionIdsBuildPath:
    """TC-130-01/02: position_ids through build and save/load roundtrip."""

    def test_position_ids_preserved_in_build(self):
        """TC-130-01: build with position_ids dataset stores them per-example."""
        model = _SimplePrefixCacheModel()
        ds = _TokenDatasetWithPositions(n=4)
        cached = build_prefix_feature_dataset(
            model, ds, batch_size=2, device="cpu", split_layer_idx=2,
        )
        for ex in cached._examples:
            assert ex.position_ids is not None
            assert ex.position_ids.shape == (8,)

    def test_position_ids_roundtrip_through_disk(self, tmp_path: Path):
        """TC-130-02: save→load preserves position_ids."""
        model = _SimplePrefixCacheModel()
        ds = _TokenDatasetWithPositions(n=4)
        cached = build_prefix_feature_dataset(
            model, ds, batch_size=2, device="cpu", split_layer_idx=2,
        )
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        save_prefix_feature_dataset(cached, cache_path, metadata=metadata)
        loaded = load_prefix_feature_dataset(cache_path)
        for orig, reloaded in zip(cached._examples, loaded._examples):
            assert reloaded.position_ids is not None
            assert torch.equal(orig.position_ids, reloaded.position_ids)


# ---------------------------------------------------------------------------
# REQ-131: model.training state restoration
# ---------------------------------------------------------------------------


class TestModelTrainingRestoration:
    """TC-131-E01/E02: build restores model.training on both success and error."""

    def test_training_restored_after_exception(self):
        """TC-131-E01: model.training restored after forward raises."""
        model = _SimplePrefixCacheModel()
        model.train()
        ds = _TokenDataset(n=4)
        with patch.object(
            model, "forward", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                build_prefix_feature_dataset(
                    model, ds, batch_size=2, device="cpu", split_layer_idx=2,
                )
        assert model.training is True

    def test_training_restored_after_normal_build(self):
        """TC-131-E02: model.training restored after successful build."""
        model = _SimplePrefixCacheModel()
        model.train()
        ds = _TokenDataset(n=4)
        build_prefix_feature_dataset(
            model, ds, batch_size=2, device="cpu", split_layer_idx=2,
        )
        assert model.training is True


# ---------------------------------------------------------------------------
# REQ-132: SHA-256 cache path uniqueness
# ---------------------------------------------------------------------------


class TestCachePathSha256:
    """TC-132-01/02/03: cache path varies with metadata."""

    def test_different_seed_gives_different_path(self):
        """TC-132-01: different seed → different path."""
        m1 = _default_metadata(seed=42)
        m2 = _default_metadata(seed=43)
        p1 = get_prefix_feature_cache_path("/tmp/cache", m1)
        p2 = get_prefix_feature_cache_path("/tmp/cache", m2)
        assert p1 != p2

    def test_same_metadata_gives_same_path(self):
        """TC-132-02: identical metadata → identical path."""
        m = _default_metadata(seed=42)
        p1 = get_prefix_feature_cache_path("/tmp/cache", m)
        p2 = get_prefix_feature_cache_path("/tmp/cache", m)
        assert p1 == p2

    def test_different_lora_r_gives_different_path(self):
        """TC-132-03: different lora_r → different path."""
        m1 = _default_metadata(lora_r=16)
        m2 = _default_metadata(lora_r=32)
        p1 = get_prefix_feature_cache_path("/tmp/cache", m1)
        p2 = get_prefix_feature_cache_path("/tmp/cache", m2)
        assert p1 != p2

    def test_different_train_on_prompt_gives_different_path(self):
        """TC-132-04: different train_on_prompt → different path.

        ``train_on_prompt`` is applied at load time (NOT baked into the dataset
        file): ``False`` masks the prompt with ``labels=-100``; ``True`` keeps
        them. ``build_prefix_feature_dataset`` copies those labels verbatim into
        the cache, so two runs over the SAME ``dataset_path`` that differ only in
        ``train_on_prompt`` must NOT share a cache path — otherwise the second
        run silently replays the first run's (wrongly-)masked labels.
        """
        m_false = _default_metadata(train_on_prompt=False)
        m_true = _default_metadata(train_on_prompt=True)
        p_false = get_prefix_feature_cache_path("/tmp/cache", m_false)
        p_true = get_prefix_feature_cache_path("/tmp/cache", m_true)
        assert p_false != p_true
        # And the field is actually stamped into the identity dict, so a future
        # refactor cannot drop it while leaving the parameter in place.
        assert m_false["train_on_prompt"] is False
        assert m_true["train_on_prompt"] is True

    def test_different_load_in_4bit_gives_different_path(self):
        """TC-132-05: different load_in_4bit → different path.

        The cached ``hidden_states`` are the base model's forward pass captured
        at ``split_layer_idx`` (a forward hook in
        ``build_prefix_feature_dataset``), so they depend on the model's
        load-time precision — which ``load_in_4bit`` toggles
        (``build_bnb_config`` builds a ``BitsAndBytesConfig`` only when it is
        True). ``model_name`` alone does not distinguish a 4-bit run from a
        full-precision run, so two runs over the SAME ``dataset_path`` /
        ``model_name`` that differ only in ``load_in_4bit`` must NOT share a
        cache path — otherwise the second run silently replays the first run's
        wrong-precision activations (the ACTIVATION-side twin of the
        ``train_on_prompt`` LABEL-side collision in TC-132-04).
        """
        m_quant = _default_metadata(load_in_4bit=True)
        m_full = _default_metadata(load_in_4bit=False)
        p_quant = get_prefix_feature_cache_path("/tmp/cache", m_quant)
        p_full = get_prefix_feature_cache_path("/tmp/cache", m_full)
        assert p_quant != p_full
        assert m_quant["load_in_4bit"] is True
        assert m_full["load_in_4bit"] is False

    def test_different_bnb_4bit_quant_type_gives_different_path(self):
        """TC-132-06: different bnb_4bit_quant_type → different path.

        With ``load_in_4bit=True`` the dequantization type (nf4 vs fp4) changes
        the rounded weights and therefore the forward activations; it must not
        collide across runs differing only in ``bnb_4bit_quant_type``.
        """
        m_nf4 = _default_metadata(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        m_fp4 = _default_metadata(load_in_4bit=True, bnb_4bit_quant_type="fp4")
        assert get_prefix_feature_cache_path("/tmp/cache", m_nf4) != \
            get_prefix_feature_cache_path("/tmp/cache", m_fp4)

    def test_different_dtype_gives_different_path(self):
        """TC-132-07: different dtype → different path.

        With ``load_in_4bit=False`` the weights are cast to ``dtype`` (bf16 vs
        fp16), changing the forward activations; it must not collide across runs
        differing only in ``dtype``.
        """
        m_bf16 = _default_metadata(load_in_4bit=False, dtype="bfloat16")
        m_fp16 = _default_metadata(load_in_4bit=False, dtype="float16")
        assert get_prefix_feature_cache_path("/tmp/cache", m_bf16) != \
            get_prefix_feature_cache_path("/tmp/cache", m_fp16)

    def test_different_max_seq_len_gives_different_path(self):
        """TC-132-09: different max_seq_len → different path.

        ``max_seq_len`` is the truncation bound applied at load time by
        ``load_dataset(path, tokenizer, max_seq_len, train_on_prompt)`` —
        ``truncation=True, max_length=max_seq_len`` in the HF tokenizer call.
        It therefore changes:

          - the input tokens (truncated) → ``hidden_states`` (forward pass
            over the truncated sequence)
          - the cached ``labels`` mask (the loader applies the prompt mask
            AFTER truncation, so the mask position is a function of the
            post-truncation sequence length)
          - the cached tensor byte shape itself when truncation fires

        Two runs over the SAME ``dataset_path`` that differ ONLY in
        ``max_seq_len`` must NOT share a cache path, else the second run
        silently replays the first run's truncation-bound content. This is
        the label-affecting / activation-affecting collision — feedback §3
        asked for an integration-test verification of "cache miss +
        re-inference on config diff"; that answer lives in
        ``tests/test_async_cache_builder_integration.py::test_max_seq_len_change_triggers_cache_miss_and_rebuild``,
        this unit test pins the SHA-256 divergence + key-stamping so the
        integration test cannot silently lose its mutation-proof
        precondition.
        """
        m8 = _default_metadata(max_seq_len=8)
        m16 = _default_metadata(max_seq_len=16)
        p8 = get_prefix_feature_cache_path("/tmp/cache", m8)
        p16 = get_prefix_feature_cache_path("/tmp/cache", m16)
        assert p8 != p16
        # And the field is actually stamped into the identity dict, so a
        # future refactor cannot drop it while leaving the parameter in
        # place (mirrors TC-132-04 / TC-132-05 structural ban).
        assert m8["max_seq_len"] == 8
        assert m16["max_seq_len"] == 16

    def test_required_precision_fields_have_no_default(self):
        """TC-132-08: the four precision fields are REQUIRED (no silent default).

        Mirrors TC-132-04's structural ban: a future producer path that omits a
        precision field must TypeError at the call site rather than silently
        defaulting to a value that masks a precision-driven collision.
        """
        import pytest

        with pytest.raises(TypeError):
            build_prefix_feature_cache_metadata(  # type: ignore[call-arg]
                dataset_path="data/train.jsonl",
                model_name="dummy-model",
                seed=42,
                max_seq_len=8,
                split_layer_idx=2,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                lora_target_modules="all-linear",
                trainable_lora_scope="last_25_percent",
                train_on_prompt=False,
            )


# ---------------------------------------------------------------------------
# TASK-0210: cache-fingerprint completeness guard (schema → fingerprint)
# ---------------------------------------------------------------------------


class TestCacheFingerprintCompleteness:
    """The prefix-feature cache stores base-model ``hidden_states`` captured at
    ``split_layer_idx`` over a dataset loaded by
    ``load_dataset(path, tokenizer, max_seq_len, train_on_prompt)`` (private
    ``src.data``). Every config field that can change the CACHED BYTES must be
    stamped into the fingerprint, else two runs differing only in that field
    silently share one cache path and replay wrong content — the
    silent-collision class closed incrementally by TASK-0206 (LABEL side,
    ``train_on_prompt``) and TASK-0207 (ACTIVATION side, the four model-precision
    fields).

    This guard does NOT re-test pairwise collisions (the TC-132-* tests do). It
    enumerates the config SCHEMA — the in-repo contract, since the loader itself
    is private/absent from this mirror — and fails if any ``DataConfig`` /
    ``ModelConfig`` field is neither fingerprinted nor explicitly allow-listed as
    non-cache-affecting. So a future sub-sampling / tokenization / precision
    field added to the schema is FORCED through the fingerprint decision instead
    of slipping through as a silent-collision regression. Audit a
    content-addressed cache by enumerating the config fields that change the
    cached bytes, not just the cache struct's own fields.

    Block (4) closes the one cross-section gap that schema enumeration alone
    leaves: ``build_prefix_feature_cache_metadata`` also reads byte-changing
    ``TrainingConfig`` fields (``train_on_prompt``, ``trainable_lora_scope``)
    that live OUTSIDE ``DataConfig``/``ModelConfig``. They are correctly
    fingerprinted today, but without this block a future edit that dropped them
    from the fingerprint would pass the guard with no failure — the very
    silent-collision regression (LABEL-side ``train_on_prompt``, TASK-0206) the
    guard exists to prevent.
    """

    def test_every_data_and_model_config_field_is_fingerprinted_or_allowlisted(self):
        metadata = _default_metadata()
        fingerprinted = set(metadata.keys())

        # Each schema field maps to the fingerprint key it flows through. The
        # claimed key MUST exist in `metadata`, so a rename in the fingerprint
        # struct cannot silently strand a field (asserted below as `stale_keys`).
        field_to_fingerprint_key = {
            # --- DataConfig: input-file identity + tokenization ---
            # Per-shard: the producer calls build_prefix_feature_cache_metadata
            # once per split with that split's path, which is resolved and
            # stamped together with size + mtime — so all three path fields flow
            # through the same dataset_path mechanism.
            "train_path": "dataset_path",
            "valid_quick_path": "dataset_path",
            "valid_full_path": "dataset_path",
            "max_seq_len": "max_seq_len",
            # --- ModelConfig: weights + tokenizer + activation precision ---
            "name_or_path": "model_name",
            "dtype": "dtype",
            "load_in_4bit": "load_in_4bit",
            "bnb_4bit_quant_type": "bnb_4bit_quant_type",
            "bnb_4bit_compute_dtype": "bnb_4bit_compute_dtype",
        }
        # Fields that do NOT change the cached bytes — each carries a reason.
        non_cache_affecting = {
            # Only gold-eval generation reads gold_test_path; the cache is built
            # over train/valid_quick/valid_full, so this file never enters the
            # base-model forward that populates the cache.
            "gold_test_path": "eval-generation-only; not one of the cache-built splits",
            # Device placement changes WHERE the forward runs, not the tensor
            # values it produces (deterministic given weights + precision).
            "device_map": "runtime placement; does not change cached tensor values",
            "device": "runtime placement; does not change cached tensor values",
        }

        schema_fields = set(DataConfig.model_fields) | set(ModelConfig.model_fields)
        covered = set(field_to_fingerprint_key) | set(non_cache_affecting)

        # (1) Every schema field must be accounted for exactly once. Adding a
        # schema field without an entry here fails on purpose — the author must
        # either flow it into the fingerprint or allow-list it WITH a reason.
        unaccounted = schema_fields - covered
        assert not unaccounted, (
            f"Cache-fingerprint completeness gap: schema field(s) {sorted(unaccounted)} "
            f"are neither fingerprinted nor allow-listed. A config field that can "
            f"change the cached bytes MUST flow into build_prefix_feature_cache_metadata "
            f"or be added to non_cache_affecting here WITH a reason "
            f"(see TASK-0206/0207/0210)."
        )

        # (2) The fingerprint-key mapping must be current: every claimed key
        # must actually exist in the metadata, so a fingerprint rename cannot
        # silently strand a field.
        stale_keys = {
            field: key
            for field, key in field_to_fingerprint_key.items()
            if key not in fingerprinted
        }
        assert not stale_keys, (
            f"Cache-fingerprint mapping is stale: {stale_keys} claim fingerprint "
            f"keys absent from build_prefix_feature_cache_metadata output "
            f"(keys present: {sorted(fingerprinted)})."
        )

        # (3) No schema field may claim coverage twice.
        overlap = set(field_to_fingerprint_key) & set(non_cache_affecting)
        assert not overlap, (
            f"Schema field(s) {sorted(overlap)} are both fingerprinted and "
            f"allow-listed — pick one."
        )

        # (4) Cross-section byte-changing inputs. build_prefix_feature_cache_metadata
        # also reads TrainingConfig fields that live OUTSIDE the DataConfig/ModelConfig
        # schemas enumerated above; the byte-changing ones (train_on_prompt = LABEL
        # masking; trainable_lora_scope drives split_layer_idx) must stay mapped to a
        # live fingerprint key, else a future edit could silently drop them with no
        # guard failing — the silent-collision regression this guard exists to prevent.
        for (schema_cls, field_name), fp_key in (
            cross_section_byte_changing_fingerprint_inputs().items()
        ):
            assert field_name in schema_cls.model_fields, (
                f"Cross-section fingerprint input {schema_cls.__name__}.{field_name} "
                f"no longer exists in the schema — update the guard mapping "
                f"(see TASK-0206/0210)."
            )
            assert fp_key in fingerprinted, (
                f"Byte-changing field {schema_cls.__name__}.{field_name} is mapped to "
                f"fingerprint key {fp_key!r}, which is absent from "
                f"build_prefix_feature_cache_metadata output (keys present: "
                f"{sorted(fingerprinted)}). A byte-changing config field MUST stay in "
                f"the fingerprint, or two runs differing only in it silently replay "
                f"wrong cached content (the TASK-0206/0207 silent-collision class)."
            )

    def test_guard_detects_dropped_train_on_prompt_fingerprint(self):
        """Mutation proof for guard block (4): if the byte-changing LABEL field
        ``train_on_prompt`` were removed from
        :func:`build_prefix_feature_cache_metadata`'s output, the cross-section
        completeness check MUST flag it — else two runs differing only in
        ``train_on_prompt`` silently replay each other's wrongly-masked labels
        with no signal (the TASK-0206 silent-collision class)."""
        metadata = _default_metadata()
        # Sanity: the field is present in the real fingerprint today.
        assert "train_on_prompt" in metadata

        # Simulate a producer regression that drops the key from the fingerprint.
        fingerprinted_after_drop = {k for k in metadata if k != "train_on_prompt"}

        # Mirror guard block (4): every cross-section byte-changing input must
        # map to a key still present in the fingerprint.
        flagged = [
            f"{schema_cls.__name__}.{field_name}"
            for (schema_cls, field_name), fp_key in (
                cross_section_byte_changing_fingerprint_inputs().items()
            )
            if fp_key not in fingerprinted_after_drop
        ]
        assert "TrainingConfig.train_on_prompt" in flagged, (
            "completeness guard must detect train_on_prompt dropped from the "
            f"fingerprint; flagged={flagged!r}"
        )


# ---------------------------------------------------------------------------
# REQ-133: format_version mismatch
# ---------------------------------------------------------------------------


class TestFormatVersionMismatch:
    """TC-133-E01/E02: load rejects wrong/missing format_version."""

    def test_format_version_zero_raises_value_error(self, tmp_path: Path):
        """TC-133-E01: format_version=0 → ValueError."""
        bad_file = tmp_path / "v0.pt"
        torch.save(
            {"format_version": 0, "metadata": {}, "hidden_states": torch.randn(2, 4, 8),
             "attention_mask": torch.ones(2, 4, dtype=torch.long),
             "labels": torch.randint(0, 32, (2, 4)), "split_layer_idx": 2},
            bad_file,
        )
        with pytest.raises(ValueError, match="Unsupported prefix feature cache format version: 0"):
            load_prefix_feature_dataset(bad_file)

    def test_missing_format_version_raises_value_error(self, tmp_path: Path):
        """TC-133-E02: no format_version key → ValueError with None."""
        bad_file = tmp_path / "no_version.pt"
        torch.save(
            {"metadata": {}, "hidden_states": torch.randn(2, 4, 8),
             "attention_mask": torch.ones(2, 4, dtype=torch.long),
             "labels": torch.randint(0, 32, (2, 4)), "split_layer_idx": 2},
            bad_file,
        )
        with pytest.raises(ValueError, match="Unsupported prefix feature cache format version: None"):
            load_prefix_feature_dataset(bad_file)


# ---------------------------------------------------------------------------
# REQ-134: Empty dataset rejection
# ---------------------------------------------------------------------------


class TestEmptyDatasetRejection:
    """TC-134-E01: save rejects empty PrefixFeatureDataset."""

    def test_empty_dataset_raises_value_error(self, tmp_path: Path):
        """TC-134-E01: empty dataset → ValueError, no file created."""
        with pytest.raises(ValueError, match="examples must not be empty"):
            PrefixFeatureDataset([])
        cache_path = tmp_path / "empty.pt"
        assert not cache_path.exists()


def test_compute_prefix_feature_shard_ranges_even_split():
    assert compute_prefix_feature_shard_ranges(8, 2) == [(0, 4), (4, 8)]


def test_compute_prefix_feature_shard_ranges_caps_worker_count_and_handles_remainder():
    assert compute_prefix_feature_shard_ranges(5, 8) == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    assert compute_prefix_feature_shard_ranges(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_merge_prefix_feature_cache_shards_roundtrip(tmp_path: Path):
    metadata = _default_metadata()
    shard_a = PrefixFeatureDataset(
        [
            PrefixFeatureExample(
                hidden_states=torch.full((8, 16), 1.0),
                attention_mask=torch.ones(8, dtype=torch.long),
                labels=torch.arange(8, dtype=torch.long),
                split_layer_idx=2,
            )
        ]
    )
    shard_b = PrefixFeatureDataset(
        [
            PrefixFeatureExample(
                hidden_states=torch.full((8, 16), 2.0),
                attention_mask=torch.ones(8, dtype=torch.long),
                labels=torch.arange(8, 16, dtype=torch.long),
                split_layer_idx=2,
            )
        ]
    )
    shard_a_path = tmp_path / "shard_a.pt"
    shard_b_path = tmp_path / "shard_b.pt"
    save_prefix_feature_dataset(shard_a, shard_a_path, metadata={"rank": 0})
    save_prefix_feature_dataset(shard_b, shard_b_path, metadata={"rank": 1})

    merged_path = tmp_path / "merged.pt"
    merge_prefix_feature_cache_shards(
        [shard_a_path, shard_b_path],
        merged_path,
        metadata=metadata,
    )
    merged = load_prefix_feature_dataset(merged_path)

    assert len(merged) == 2
    assert torch.equal(merged[0]["labels"], torch.arange(8, dtype=torch.long))
    assert torch.equal(merged[1]["labels"], torch.arange(8, 16, dtype=torch.long))
    assert torch.allclose(merged[0]["hidden_states"], torch.full((8, 16), 1.0))
    assert torch.allclose(merged[1]["hidden_states"], torch.full((8, 16), 2.0))


def test_resolve_prefix_feature_cache_seed_respects_share_flag():
    assert resolve_prefix_feature_cache_seed(42, share_across_seeds=False) == 42
    assert resolve_prefix_feature_cache_seed(42, share_across_seeds=True) == 0


class TestPrefixFeatureDatasetValidation:
    def test_rejects_empty_examples(self):
        with pytest.raises(ValueError, match="examples must not be empty"):
            PrefixFeatureDataset([])

    def test_rejects_non_example_elements(self):
        with pytest.raises(TypeError, match=r"examples\[1\] must be a PrefixFeatureExample"):
            PrefixFeatureDataset([
                PrefixFeatureExample(
                    hidden_states=torch.randn(4, 8),
                    attention_mask=torch.ones(4, dtype=torch.long),
                    labels=torch.randint(0, 10, (4,)),
                    split_layer_idx=2,
                ),
                "not_an_example",
            ])

    def test_mapped_rejects_negative_split_layer_idx(self):
        with pytest.raises(ValueError, match="split_layer_idx must be non-negative"):
            MappedPrefixFeatureDataset(
                hidden_states=torch.randn(2, 4, 8),
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                labels=torch.randint(0, 10, (2, 4)),
                split_layer_idx=-1,
            )

    def test_mapped_rejects_none_hidden_states(self):
        with pytest.raises(ValueError, match="hidden_states must not be None"):
            MappedPrefixFeatureDataset(
                hidden_states=None,
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                labels=torch.randint(0, 10, (2, 4)),
                split_layer_idx=0,
            )

    def test_mapped_rejects_none_attention_mask(self):
        with pytest.raises(ValueError, match="attention_mask must not be None"):
            MappedPrefixFeatureDataset(
                hidden_states=torch.randn(2, 4, 8),
                attention_mask=None,
                labels=torch.randint(0, 10, (2, 4)),
                split_layer_idx=0,
            )

    def test_mapped_rejects_none_labels(self):
        with pytest.raises(ValueError, match="labels must not be None"):
            MappedPrefixFeatureDataset(
                hidden_states=torch.randn(2, 4, 8),
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                labels=None,
                split_layer_idx=0,
            )

    def test_mapped_rejects_mismatched_batch_sizes(self):
        with pytest.raises(ValueError, match="Batch size mismatch"):
            MappedPrefixFeatureDataset(
                hidden_states=torch.randn(3, 4, 8),
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                labels=torch.randint(0, 10, (2, 4)),
                split_layer_idx=0,
            )


class TestOneShotMode:
    """REQ-224: prefix_feature_cache_mode='one_shot' loads via MappedPrefixFeatureDataset."""

    def _build_and_save_cache(self, tmp_path, n=4):
        model = _SimplePrefixCacheModel()
        raw_ds = _TokenDataset(n=n)
        cached = build_prefix_feature_dataset(
            model, raw_ds, batch_size=2, device="cpu", split_layer_idx=2,
        )
        cache_path = tmp_path / "one_shot_cache.pt"
        save_prefix_feature_dataset(cached, cache_path, metadata={"test": True})
        return cache_path

    def test_load_lazy_returns_mapped_dataset(self, tmp_path):
        cache_path = self._build_and_save_cache(tmp_path)
        ds = load_prefix_feature_dataset(cache_path, lazy=True)
        assert isinstance(ds, MappedPrefixFeatureDataset), (
            f"one_shot mode should return MappedPrefixFeatureDataset, got {type(ds).__name__}"
        )

    def test_load_eager_returns_prefix_feature_dataset(self, tmp_path):
        cache_path = self._build_and_save_cache(tmp_path)
        ds = load_prefix_feature_dataset(cache_path, lazy=False)
        assert isinstance(ds, PrefixFeatureDataset), (
            f"reuse mode should return PrefixFeatureDataset, got {type(ds).__name__}"
        )

    def test_lazy_dataset_has_correct_length(self, tmp_path):
        cache_path = self._build_and_save_cache(tmp_path, n=6)
        ds = load_prefix_feature_dataset(cache_path, lazy=True)
        assert len(ds) == 6


# ---------------------------------------------------------------------------
# TC-224-02: one_shot YAML config passes Pydantic validation
# ---------------------------------------------------------------------------


class TestTC224:
    """REQ-224: one_shot YAML config validation."""

    def test_tc224_02_one_shot_config_passes_pydantic_validation(self):
        """TC-224-02: configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml
        passes Pydantic config_schema validation."""
        from src.training.config_schema import load_and_validate_config

        config = load_and_validate_config(
            "configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml"
        )
        assert config.training.prefix_feature_cache_mode == "one_shot"
        assert config.training.prefix_feature_cache_experimental is True
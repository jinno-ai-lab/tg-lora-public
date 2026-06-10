"""Unit tests for src/data/build_seed_dataset.py (LoraDataset, load_dataset)."""

from pathlib import Path

import orjson
import pytest
import torch
from transformers import AutoTokenizer

from src.data.build_seed_dataset import (LoraDataset,
                                         analyze_record_supervision,
                                         load_dataset)

pytestmark = pytest.mark.network

MAX_SEQ = 64


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def _records(n: int = 5) -> list[dict]:
    return [{"text": f"hello world example number {i}"} for i in range(n)]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


# ── __len__ ──────────────────────────────────────────────────────────────────


class TestLoraDatasetLen:
    def test_len_matches_record_count(self, tokenizer):
        ds = LoraDataset(_records(7), tokenizer, MAX_SEQ)
        assert len(ds) == 7

    def test_len_zero_for_empty_records(self, tokenizer):
        with pytest.raises(ValueError, match="records must be a non-empty list"):
            LoraDataset([], tokenizer, MAX_SEQ)

    def test_len_single_record(self, tokenizer):
        ds = LoraDataset(_records(1), tokenizer, MAX_SEQ)
        assert len(ds) == 1


# ── __getitem__ ──────────────────────────────────────────────────────────────


class TestLoraDatasetGetitem:
    def test_returns_correct_keys(self, tokenizer):
        ds = LoraDataset(_records(1), tokenizer, MAX_SEQ)
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "labels", "attention_mask"}

    def test_tensors_have_correct_shape(self, tokenizer):
        ds = LoraDataset(_records(1), tokenizer, MAX_SEQ)
        item = ds[0]
        assert item["input_ids"].shape == (MAX_SEQ,)
        assert item["attention_mask"].shape == (MAX_SEQ,)
        assert item["labels"].shape == (MAX_SEQ,)

    def test_tensors_are_long_dtype(self, tokenizer):
        ds = LoraDataset(_records(1), tokenizer, MAX_SEQ)
        item = ds[0]
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long
        assert item["labels"].dtype == torch.long

    def test_labels_mask_padding_with_neg100(self, tokenizer):
        short_text = "hi"
        ds = LoraDataset([{"text": short_text}], tokenizer, MAX_SEQ)
        item = ds[0]
        pad_id = tokenizer.pad_token_id
        pad_positions = item["input_ids"] == pad_id
        assert torch.all(item["labels"][pad_positions] == -100)

    def test_labels_match_input_ids_where_not_padding(self, tokenizer):
        ds = LoraDataset(_records(1), tokenizer, MAX_SEQ)
        item = ds[0]
        non_pad = item["labels"] != -100
        assert torch.all(item["input_ids"][non_pad] == item["labels"][non_pad])

    def test_uses_prompt_and_completion_when_no_text(self, tokenizer):
        rec = {"prompt": "question: ", "completion": "answer"}
        ds = LoraDataset([rec], tokenizer, MAX_SEQ)
        item = ds[0]
        full = tokenizer(
            "question: answer",
            max_length=MAX_SEQ,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        assert torch.equal(item["input_ids"], full["input_ids"].squeeze(0))

    def test_masks_prompt_tokens_when_prompt_completion_record(self, tokenizer):
        rec = {"prompt": "question: ", "completion": "answer"}
        ds = LoraDataset([rec], tokenizer, MAX_SEQ, train_on_prompt=False)
        item = ds[0]
        full = tokenizer(
            rec["prompt"] + rec["completion"],
            max_length=MAX_SEQ,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        boundary = len(rec["prompt"].rstrip())
        offsets = full["offset_mapping"].squeeze(0)
        masked = (item["attention_mask"] == 1) & (offsets[:, 1] <= boundary)
        unmasked = (item["attention_mask"] == 1) & (offsets[:, 1] > boundary)
        assert torch.all(item["labels"][masked] == -100)
        assert torch.any(item["labels"][unmasked] != -100)

    def test_masks_chatml_prefix_before_assistant_response(self, tokenizer):
        rec = {
            "text": "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld<|im_end|>"
        }
        ds = LoraDataset([rec], tokenizer, MAX_SEQ, train_on_prompt=False)
        item = ds[0]
        prompt_text = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        prompt_len = tokenizer(
            prompt_text,
            max_length=MAX_SEQ,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )["input_ids"].shape[-1]
        assert torch.all(item["labels"][:prompt_len] == -100)
        assert torch.any(item["labels"][prompt_len:] != -100)

    def test_train_on_prompt_keeps_prompt_labels(self, tokenizer):
        rec = {"prompt": "question: ", "completion": "answer"}
        ds = LoraDataset([rec], tokenizer, MAX_SEQ, train_on_prompt=True)
        item = ds[0]
        non_pad = item["attention_mask"] == 1
        assert torch.all(item["labels"][non_pad] == item["input_ids"][non_pad])

    def test_indexing_multiple_items(self, tokenizer):
        ds = LoraDataset(_records(3), tokenizer, MAX_SEQ)
        for i in range(3):
            item = ds[i]
            assert item["input_ids"].shape == (MAX_SEQ,)


# ── load_dataset ─────────────────────────────────────────────────────────────


class TestLoadDataset:
    def test_loads_jsonl_and_returns_lora_dataset(self, tokenizer, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, _records(4))
        ds = load_dataset(str(path), tokenizer, MAX_SEQ)
        assert isinstance(ds, LoraDataset)
        assert len(ds) == 4

    def test_loaded_items_are_valid(self, tokenizer, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, _records(2))
        ds = load_dataset(str(path), tokenizer, MAX_SEQ)
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "labels", "attention_mask"}
        assert item["input_ids"].shape == (MAX_SEQ,)

    def test_load_with_validate_true(self, tokenizer, tmp_path):
        path = tmp_path / "data.jsonl"
        records = [
            {"text": "<|im_start|>user\nhello<|im_end|>", "token_count": 5},
            {"text": "<|im_start|>assistant\nhi there<|im_end|>", "token_count": 8},
        ]
        _write_jsonl(path, records)
        ds = load_dataset(str(path), tokenizer, MAX_SEQ, validate=True)
        assert isinstance(ds, LoraDataset)
        assert len(ds) == 2

    def test_load_with_validate_skips_invalid(self, tokenizer, tmp_path):
        path = tmp_path / "data.jsonl"
        records = [
            {"text": "<|im_start|>valid<|im_end|>", "token_count": 3},
            {"text": "", "token_count": 0},
        ]
        _write_jsonl(path, records)
        ds = load_dataset(str(path), tokenizer, MAX_SEQ, validate=True)
        assert isinstance(ds, LoraDataset)
        assert len(ds) == 1

    def test_load_dataset_filters_all_masked_records(self, tokenizer, tmp_path):
        path = tmp_path / "data.jsonl"
        masked = {"prompt": "A " * 200, "completion": "Z"}
        valid = {"prompt": "A ", "completion": "Z"}
        assert analyze_record_supervision(masked, tokenizer, 16)["all_masked"] is True
        assert analyze_record_supervision(valid, tokenizer, 16)["all_masked"] is False
        _write_jsonl(path, [masked, valid])

        ds = load_dataset(str(path), tokenizer, 16)

        assert isinstance(ds, LoraDataset)
        assert len(ds) == 1
        assert ds.records[0]["completion"] == "Z"


# ── error handling: empty / invalid JSONL ────────────────────────────────────


class TestEdgeCases:
    def test_empty_file_raises_value_error(self, tokenizer, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="records must be a non-empty list"):
            load_dataset(str(path), tokenizer, MAX_SEQ)

    def test_file_with_only_blank_lines_raises_value_error(self, tokenizer, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_bytes(b"  \n\n  \n")
        with pytest.raises(ValueError, match="records must be a non-empty list"):
            load_dataset(str(path), tokenizer, MAX_SEQ)

    def test_invalid_jsonl_raises_error(self, tokenizer, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_bytes(b"not valid json\n")
        with pytest.raises(Exception):
            load_dataset(str(path), tokenizer, MAX_SEQ)

    def test_mixed_valid_and_invalid_lines(self, tokenizer, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_bytes(b'{"text": "ok"}\nnot json\n')
        with pytest.raises(Exception):
            load_dataset(str(path), tokenizer, MAX_SEQ)

    def test_missing_file_raises_error(self, tokenizer, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "nonexistent.jsonl"), tokenizer, MAX_SEQ)


# ── constructor validation ────────────────────────────────────────────────────


class TestLoraDatasetValidation:
    def test_init_rejects_empty_records(self, tokenizer):
        with pytest.raises(ValueError, match="records must be a non-empty list"):
            LoraDataset([], tokenizer, MAX_SEQ)

    def test_init_rejects_none_tokenizer(self):
        with pytest.raises(ValueError, match="tokenizer must not be None"):
            LoraDataset([{"text": "hello"}], None, MAX_SEQ)

    def test_init_rejects_nonpositive_max_seq_len(self, tokenizer):
        with pytest.raises(ValueError, match="max_seq_len must be a positive int"):
            LoraDataset([{"text": "hello"}], tokenizer, 0)

    def test_init_rejects_negative_max_seq_len(self, tokenizer):
        with pytest.raises(ValueError, match="max_seq_len must be a positive int"):
            LoraDataset([{"text": "hello"}], tokenizer, -1)

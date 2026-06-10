"""Unit tests for src/data/filter_dataset.py."""

import orjson
import torch

from src.data.filter_dataset import filter_dataset
from src.utils.io import load_jsonl


class _DummyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        max_length,
        truncation,
        padding,
        return_tensors,
        return_offsets_mapping=False,
    ):
        del truncation, return_tensors
        text = text[:max_length]
        token_count = len(text)
        ids = list(range(1, token_count + 1))
        attention_mask = [1] * token_count
        offsets = [(idx, idx + 1) for idx in range(token_count)]
        if padding == "max_length":
            pad = max_length - token_count
            ids.extend([0] * pad)
            attention_mask.extend([0] * pad)
            offsets.extend([(0, 0)] * pad)
        out = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        return out


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def _record(text="a" * 50, quality_score=0.8, **extra):
    rec = {"text": text, "quality_score": quality_score, **extra}
    return rec


# ── text length filtering ──────────────────────────────────────────────────


class TestTextLengthFilter:
    def test_removes_records_below_min_length(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [_record(text="short")])
        filter_dataset(str(inp), str(out), min_length=20)
        assert load_jsonl(str(out)) == []

    def test_removes_records_above_max_length(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [_record(text="a" * 5000)])
        filter_dataset(str(inp), str(out), max_length=4096)
        assert load_jsonl(str(out)) == []

    def test_keeps_records_within_length_range(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record(text="a" * 100)
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), min_length=20, max_length=4096)
        result = load_jsonl(str(out))
        assert len(result) == 1

    def test_min_length_boundary_kept(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record(text="a" * 20)
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), min_length=20)
        assert len(load_jsonl(str(out))) == 1

    def test_max_length_boundary_kept(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record(text="a" * 4096)
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), max_length=4096)
        assert len(load_jsonl(str(out))) == 1


# ── quality score filtering ────────────────────────────────────────────────


class TestQualityScoreFilter:
    def test_removes_records_below_min_quality(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [_record(quality_score=0.3)])
        filter_dataset(str(inp), str(out), min_quality_score=0.5)
        assert load_jsonl(str(out)) == []

    def test_keeps_records_at_or_above_min_quality(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record(quality_score=0.5)
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), min_quality_score=0.5)
        assert len(load_jsonl(str(out))) == 1

    def test_reads_quality_from_provenance_subdict(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record(quality_score=None)
        rec.pop("quality_score")
        rec["provenance"] = {"quality_score": 0.9}
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), min_quality_score=0.5)
        assert len(load_jsonl(str(out))) == 1

    def test_non_numeric_quality_defaults_to_zero(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record(quality_score="bad")
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), min_quality_score=0.5)
        assert load_jsonl(str(out)) == []

    def test_default_quality_score_accepts_zero(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = {"text": "a" * 50}
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), min_quality_score=0.0)
        assert len(load_jsonl(str(out))) == 1


# ── required fields ────────────────────────────────────────────────────────


class TestRequiredFields:
    def test_removes_records_missing_required_field(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record()
        del rec["text"]
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), required_fields=["text"])
        assert load_jsonl(str(out)) == []

    def test_keeps_records_with_all_required_fields(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record(label="positive")
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), required_fields=["text", "label"])
        assert len(load_jsonl(str(out))) == 1

    def test_empty_required_field_excluded(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record(label="")
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out), required_fields=["label"])
        assert load_jsonl(str(out)) == []

    def test_default_required_fields_is_text(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _record()
        _write_jsonl(inp, [rec])
        filter_dataset(str(inp), str(out))
        assert len(load_jsonl(str(out))) == 1


# ── combined / edge cases ──────────────────────────────────────────────────


class TestCombinedFilters:
    def test_multiple_records_mixed(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        records = [
            _record(text="a" * 100, quality_score=0.9),
            _record(text="short", quality_score=0.9),
            _record(text="a" * 100, quality_score=0.1),
            _record(text="a" * 100, quality_score=0.7),
        ]
        _write_jsonl(inp, records)
        filter_dataset(str(inp), str(out), min_length=20, min_quality_score=0.5)
        result = load_jsonl(str(out))
        assert len(result) == 2

    def test_empty_input_produces_empty_output(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [])
        filter_dataset(str(inp), str(out))
        assert load_jsonl(str(out)) == []

    def test_all_filtered_out(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [_record(text="x"), _record(text="yy")])
        filter_dataset(str(inp), str(out), min_length=100)
        assert load_jsonl(str(out)) == []


class TestTokenAwareFiltering:
    def test_removes_all_masked_records(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        records = [
            {
                "text": "<|im_start|>user\nABCDEFGHIJK<|im_end|>\n<|im_start|>assistant\nZ<|im_end|>",
                "completion": "Z",
            },
            {
                "text": "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\nZ<|im_end|>",
                "completion": "Z",
            },
        ]
        _write_jsonl(inp, records)

        summary = filter_dataset(
            str(inp),
            str(out),
            min_length=0,
            tokenizer=_DummyTokenizer(),
            max_seq_len=60,
            min_supervised_tokens=1,
            required_fields=["text", "completion"],
        )

        result = load_jsonl(str(out))
        assert len(result) == 1
        assert summary["removed_below_min_supervised_tokens"] == 1
        assert summary["all_masked_records"] == 1

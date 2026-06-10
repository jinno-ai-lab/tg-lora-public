import json

import torch

from scripts.prepare_data import format_record, prepare_dataset
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


def test_chatml_format_without_context():
    record = {"instruction": "What is 2+2?", "response": "4"}
    result = format_record(record)

    expected = (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>"
    )
    assert result["text"] == expected
    assert result["prompt"] == "What is 2+2?"
    assert result["completion"] == "4"


def test_chatml_format_with_context():
    record = {
        "instruction": "Summarize",
        "context": "Some long text here",
        "response": "Short summary",
    }
    result = format_record(record)

    expected = (
        "<|im_start|>user\nSummarize\n\n"
        "Context: Some long text here<|im_end|>\n"
        "<|im_start|>assistant\nShort summary<|im_end|>"
    )
    assert result["text"] == expected
    assert "Context: Some long text here" in result["text"]


def test_chatml_empty_context_uses_simple_template():
    record = {"instruction": "Hello", "context": "  ", "response": "Hi"}
    result = format_record(record)

    assert "Context:" not in result["text"]
    assert (
        result["text"]
        == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>"
    )


def test_prepare_dataset_writes_token_audit_report_and_filters(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    source = raw_dir / "dolly_15k.jsonl"
    records = [
        {
            "instruction": "ABCDEFGHIJK",
            "context": "",
            "response": "Z",
        },
        {
            "instruction": "A",
            "context": "",
            "response": "Z",
        },
        {
            "instruction": "B",
            "context": "",
            "response": "Y",
        },
    ]
    with open(source, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    monkeypatch.setattr("scripts.prepare_data._load_audit_tokenizer", lambda _name: _DummyTokenizer())

    audit_report = prepare_dataset(
        source_path=source,
        output_dir=out_dir,
        train_size=2,
        valid_size=1,
        seed=0,
        audit_model_name="dummy",
        audit_max_seq_len=60,
        min_supervised_tokens=1,
    )

    assert audit_report is not None
    assert (out_dir / "token_audit_report.json").exists()
    assert (out_dir / "token_audit_report.md").exists()
    train_records = load_jsonl(str(out_dir / "train.jsonl"))
    valid_records = load_jsonl(str(out_dir / "valid_quick.jsonl"))
    assert len(train_records) + len(valid_records) == 2
    assert audit_report["splits"]["train"]["removed_below_min_supervised_tokens"] >= 0

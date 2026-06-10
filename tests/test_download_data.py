"""Tests for download_data.py: TC-025-01 (Dolly) and TC-025-02 (Capybara).

These tests mock the HuggingFace datasets library to verify JSONL conversion
logic without requiring network access.
"""

import json
from unittest.mock import patch


class FakeDataset:
    """Minimal dataset-like object that iterates over rows."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


# ---------------------------------------------------------------------------
# TC-025-01: Dolly 15k download & JSONL conversion
# ---------------------------------------------------------------------------


def _make_dolly_rows():
    return [
        {
            "instruction": "What is the capital of France?",
            "context": "Geography facts",
            "response": "Paris",
            "category": "geography",
        },
        {
            "instruction": "Explain gravity",
            "context": "",
            "response": "A force of attraction between masses",
            "category": "science",
        },
    ]


@patch("scripts.download_data.load_dataset")
def test_dolly_download_produces_jsonl(mock_load, tmp_path):
    """TC-025-01: Dolly download converts to JSONL with correct fields."""
    mock_load.return_value = FakeDataset(_make_dolly_rows())

    from scripts.download_data import download_dolly

    out_path = download_dolly(tmp_path)

    assert out_path == tmp_path / "dolly_15k.jsonl"
    assert out_path.exists()

    records = [json.loads(line) for line in out_path.read_text().strip().splitlines()]
    assert len(records) == 2

    # Verify all required fields present
    for rec in records:
        assert "instruction" in rec
        assert "context" in rec
        assert "response" in rec
        assert "category" in rec

    assert records[0]["instruction"] == "What is the capital of France?"
    assert records[0]["response"] == "Paris"
    assert records[1]["context"] == ""
    assert records[1]["category"] == "science"


@patch("scripts.download_data.load_dataset")
def test_dolly_jsonl_one_record_per_line(mock_load, tmp_path):
    """TC-025-01 variant: each line is a valid JSON object, no trailing commas."""
    mock_load.return_value = FakeDataset(_make_dolly_rows())

    from scripts.download_data import download_dolly

    out_path = download_dolly(tmp_path)
    lines = out_path.read_text().strip().splitlines()

    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert isinstance(obj, dict)


# ---------------------------------------------------------------------------
# TC-025-02: Capybara download & JSONL conversion
# ---------------------------------------------------------------------------


def _make_capybara_rows():
    return [
        {
            "conversation": [
                {"input": "Hello!", "output": "Hi there!"},
                {"input": "How are you?", "output": "Fine!"},
            ],
        },
        {
            "conversation": [
                {"input": "Explain AI", "output": "Artificial intelligence is..."},
            ],
        },
    ]


@patch("scripts.download_data.load_dataset")
def test_capybara_download_produces_jsonl(mock_load, tmp_path):
    """TC-025-02: Capybara download extracts first turn to JSONL."""
    mock_load.return_value = FakeDataset(_make_capybara_rows())

    from scripts.download_data import download_capybara

    out_path = download_capybara(tmp_path)

    assert out_path == tmp_path / "capybara.jsonl"
    assert out_path.exists()

    records = [json.loads(line) for line in out_path.read_text().strip().splitlines()]
    assert len(records) == 2

    # First turn extraction
    assert records[0]["instruction"] == "Hello!"
    assert records[0]["response"] == "Hi there!"
    assert records[0]["turns"] == 2

    assert records[1]["instruction"] == "Explain AI"
    assert records[1]["response"] == "Artificial intelligence is..."
    assert records[1]["turns"] == 1


@patch("scripts.download_data.load_dataset")
def test_capybara_skips_empty_conversations(mock_load, tmp_path):
    """TC-025-02 variant: rows with empty conversation are skipped."""
    rows = _make_capybara_rows() + [{"conversation": []}]
    mock_load.return_value = FakeDataset(rows)

    from scripts.download_data import download_capybara

    out_path = download_capybara(tmp_path)
    records = [json.loads(line) for line in out_path.read_text().strip().splitlines()]

    assert len(records) == 2  # empty conversation row skipped


@patch("scripts.download_data.load_dataset")
def test_capybara_includes_source_field(mock_load, tmp_path):
    """TC-025-02 variant: each record has source='capybara'."""
    mock_load.return_value = FakeDataset(_make_capybara_rows())

    from scripts.download_data import download_capybara

    out_path = download_capybara(tmp_path)
    records = [json.loads(line) for line in out_path.read_text().strip().splitlines()]

    for rec in records:
        assert rec["source"] == "capybara"

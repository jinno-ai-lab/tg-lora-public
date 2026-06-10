"""Tests for src/data/schema.py — DataRecord and validation helpers."""

import logging

import pytest
from pydantic import ValidationError

from src.data.schema import DataRecord, ValidationSummary, validate_records


# ── DataRecord validation ──────────────────────────────────────────────────


class TestDataRecord:
    def test_valid_record(self):
        rec = DataRecord(text="<|im_start|>user\nHello<|im_end|>")
        assert rec.text == "<|im_start|>user\nHello<|im_end|>"

    def test_missing_text_field(self):
        with pytest.raises(ValidationError):
            DataRecord()

    def test_empty_text(self):
        with pytest.raises(ValidationError):
            DataRecord(text="")

    def test_whitespace_only_text(self):
        with pytest.raises(ValidationError):
            DataRecord(text="   ")

    def test_non_chatml_text(self):
        with pytest.raises(ValidationError):
            DataRecord(text="just plain text without chatml markers")

    def test_optional_fields_absent(self):
        rec = DataRecord(text="<|im_start|>user\nHi<|im_end|>")
        assert rec.source is None
        assert rec.token_count is None

    def test_optional_fields_present(self):
        rec = DataRecord(
            text="<|im_start|>user\nHi<|im_end|>",
            source="test",
            token_count=42,
        )
        assert rec.source == "test"
        assert rec.token_count == 42

    def test_token_count_zero_rejected(self):
        with pytest.raises(ValidationError):
            DataRecord(text="<|im_start|>user\nHi<|im_end|>", token_count=0)

    def test_token_count_negative_rejected(self):
        with pytest.raises(ValidationError):
            DataRecord(text="<|im_start|>user\nHi<|im_end|>", token_count=-5)


# ── ValidationSummary ──────────────────────────────────────────────────────


class TestValidationSummary:
    def test_initial_state(self):
        s = ValidationSummary()
        assert s.total == 0
        assert s.valid == 0
        assert s.skipped == 0
        assert s.errors == []

    def test_record_valid(self):
        s = ValidationSummary()
        s.record_valid()
        assert s.total == 1
        assert s.valid == 1
        assert s.skipped == 0

    def test_record_invalid(self):
        s = ValidationSummary()
        s.record_invalid("bad record")
        assert s.total == 1
        assert s.valid == 0
        assert s.skipped == 1
        assert s.errors == ["bad record"]

    def test_log_outputs_summary(self, caplog):
        s = ValidationSummary()
        s.record_valid()
        s.record_invalid("oops")
        with caplog.at_level(logging.INFO):
            s.log()
        assert "2 total, 1 valid, 1 skipped" in caplog.text

    def test_log_outputs_warnings_for_errors(self, caplog):
        s = ValidationSummary()
        s.record_invalid("broken")
        with caplog.at_level(logging.WARNING):
            s.log()
        assert "broken" in caplog.text


# ── validate_records ───────────────────────────────────────────────────────


class TestValidateRecords:
    def test_all_valid(self):
        records = [
            {"text": "<|im_start|>user\nHello<|im_end|>"},
            {"text": "<|im_start|>system\nTest<|im_end|>", "source": "s"},
        ]
        valid, summary = validate_records(records)
        assert len(valid) == 2
        assert summary.valid == 2
        assert summary.skipped == 0

    def test_filters_invalid(self):
        records = [
            {"text": "<|im_start|>user\nHello<|im_end|>"},
            {"text": "no chatml here"},
            {"no_text_field": "oops"},
        ]
        valid, summary = validate_records(records)
        assert len(valid) == 1
        assert summary.skipped == 2

    def test_empty_list(self):
        valid, summary = validate_records([])
        assert valid == []
        assert summary.total == 0

    def test_invalid_records_logged_as_warnings(self, caplog):
        records = [{"text": "plain text"}]
        with caplog.at_level(logging.WARNING):
            validate_records(records)
        assert "Skipping invalid record" in caplog.text

    def test_validation_summary_logged(self, caplog):
        records = [{"text": "<|im_start|>user\nHi<|im_end|>"}]
        with caplog.at_level(logging.INFO):
            validate_records(records)
        assert "Validation summary" in caplog.text

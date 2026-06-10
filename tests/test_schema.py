import pytest

from src.data.schema import (
    CHATML_MARKER,
    DataRecord,
    ValidationSummary,
    validate_records,
)


# ---------------------------------------------------------------------------
# DataRecord field validators
# ---------------------------------------------------------------------------


class TestDataRecord:
    def test_valid_record(self):
        rec = DataRecord(text=f"{CHATML_MARKER}user\nHello{CHATML_MARKER}assistant\nHi")
        assert rec.text.startswith(CHATML_MARKER)
        assert rec.source is None
        assert rec.token_count is None

    def test_valid_with_all_fields(self):
        rec = DataRecord(
            text=f"{CHATML_MARKER}Hello",
            source="dolly",
            token_count=42,
        )
        assert rec.source == "dolly"
        assert rec.token_count == 42

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError, match="text must not be empty"):
            DataRecord(text="   ")

    def test_text_without_chatml_rejected(self):
        with pytest.raises(ValueError, match="text must contain"):
            DataRecord(text="Just plain text without markers")

    def test_whitespace_only_no_marker_rejected(self):
        with pytest.raises(ValueError):
            DataRecord(text="   \t\n   ")

    def test_zero_token_count_rejected(self):
        with pytest.raises(ValueError, match="token_count must be > 0"):
            DataRecord(text=f"{CHATML_MARKER}Hi", token_count=0)

    def test_negative_token_count_rejected(self):
        with pytest.raises(ValueError, match="token_count must be > 0"):
            DataRecord(text=f"{CHATML_MARKER}Hi", token_count=-5)

    def test_token_count_none_accepted(self):
        rec = DataRecord(text=f"{CHATML_MARKER}Hi", token_count=None)
        assert rec.token_count is None

    def test_positive_token_count_accepted(self):
        rec = DataRecord(text=f"{CHATML_MARKER}Hi", token_count=1)
        assert rec.token_count == 1


# ---------------------------------------------------------------------------
# ValidationSummary
# ---------------------------------------------------------------------------


class TestValidationSummary:
    def test_initial_state(self):
        vs = ValidationSummary()
        assert vs.total == 0
        assert vs.valid == 0
        assert vs.skipped == 0
        assert vs.errors == []

    def test_record_valid(self):
        vs = ValidationSummary()
        vs.record_valid()
        assert vs.total == 1
        assert vs.valid == 1
        assert vs.skipped == 0

    def test_record_invalid(self):
        vs = ValidationSummary()
        vs.record_invalid("bad data")
        assert vs.total == 1
        assert vs.valid == 0
        assert vs.skipped == 1
        assert vs.errors == ["bad data"]

    def test_mixed_records(self):
        vs = ValidationSummary()
        vs.record_valid()
        vs.record_valid()
        vs.record_invalid("err1")
        vs.record_invalid("err2")
        assert vs.total == 4
        assert vs.valid == 2
        assert vs.skipped == 2
        assert len(vs.errors) == 2


# ---------------------------------------------------------------------------
# validate_records
# ---------------------------------------------------------------------------


class TestValidateRecords:
    def test_all_valid(self):
        records = [
            {"text": f"{CHATML_MARKER}Hello"},
            {"text": f"{CHATML_MARKER}World", "source": "test"},
        ]
        valid, summary = validate_records(records)
        assert len(valid) == 2
        assert summary.total == 2
        assert summary.valid == 2
        assert summary.skipped == 0

    def test_all_invalid(self):
        records = [
            {"text": "no markers"},
            {"text": ""},
        ]
        valid, summary = validate_records(records)
        assert len(valid) == 0
        assert summary.total == 2
        assert summary.skipped == 2
        assert len(summary.errors) == 2

    def test_mixed_valid_invalid(self):
        records = [
            {"text": f"{CHATML_MARKER}Good"},
            {"text": "bad"},
            {"text": f"{CHATML_MARKER}Also good", "token_count": 10},
        ]
        valid, summary = validate_records(records)
        assert len(valid) == 2
        assert summary.total == 3
        assert summary.valid == 2
        assert summary.skipped == 1

    def test_empty_input(self):
        valid, summary = validate_records([])
        assert valid == []
        assert summary.total == 0

    def test_invalid_token_count_triggers_skip(self):
        records = [
            {"text": f"{CHATML_MARKER}Hi", "token_count": 0},
        ]
        valid, summary = validate_records(records)
        assert len(valid) == 0
        assert summary.skipped == 1

    def test_error_includes_line_number(self):
        records = [
            {"text": "no marker"},
        ]
        _, summary = validate_records(records)
        assert "line 0" in summary.errors[0]

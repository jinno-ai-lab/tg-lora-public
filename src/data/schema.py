"""Pydantic schema for ChatML training data records."""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

CHATML_MARKER = "<|im_start|>"


class DataRecord(BaseModel):
    """A single training data record in ChatML format."""

    text: str
    source: Optional[str] = None
    token_count: Optional[int] = None

    @field_validator("text")
    @classmethod
    def text_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v

    @field_validator("text")
    @classmethod
    def text_must_contain_chatml(cls, v: str) -> str:
        if CHATML_MARKER not in v:
            raise ValueError(f"text must contain {CHATML_MARKER}")
        return v

    @field_validator("token_count")
    @classmethod
    def token_count_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("token_count must be > 0")
        return v


class ValidationSummary:
    """Accumulates validation results for a batch of records."""

    def __init__(self) -> None:
        self.total: int = 0
        self.valid: int = 0
        self.skipped: int = 0
        self.errors: list[str] = []

    def record_valid(self) -> None:
        self.total += 1
        self.valid += 1

    def record_invalid(self, reason: str) -> None:
        self.total += 1
        self.skipped += 1
        self.errors.append(reason)

    def log(self) -> None:
        logger.info(
            "Validation summary: %d total, %d valid, %d skipped",
            self.total,
            self.valid,
            self.skipped,
        )
        if self.errors:
            for err in self.errors:
                logger.warning("Skipped record: %s", err)


def validate_records(records: list[dict]) -> tuple[list[dict], ValidationSummary]:
    """Validate a list of raw dicts, returning valid records and a summary."""
    summary = ValidationSummary()
    valid_records: list[dict] = []
    for i, raw in enumerate(records):
        try:
            DataRecord(**raw)
            valid_records.append(raw)
            summary.record_valid()
        except Exception as exc:
            summary.record_invalid(f"line {i}: {exc}")
            logger.warning("Skipping invalid record at index %d: %s", i, exc)
    summary.log()
    return valid_records, summary

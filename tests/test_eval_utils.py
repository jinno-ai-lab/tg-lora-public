"""Tests for shared evaluation utilities (scripts/eval_utils.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval_utils import check_json_validity, compute_char_f1, load_jsonl


class TestComputeCharF1:
    def test_identical_strings(self):
        assert compute_char_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        assert compute_char_f1("abc", "xyz") == pytest.approx(0.0)

    def test_partial_overlap(self):
        f1 = compute_char_f1("abcde", "abfgh")
        assert 0.0 < f1 < 1.0

    def test_empty_ref(self):
        assert compute_char_f1("", "hello") == 0.0

    def test_empty_gen(self):
        assert compute_char_f1("hello", "") == 0.0

    def test_both_empty(self):
        assert compute_char_f1("", "") == 0.0

    def test_whitespace_ignored(self):
        assert compute_char_f1("hello world", "helloworld") == pytest.approx(1.0)

    def test_japanese_text(self):
        f1 = compute_char_f1("今日は良い天気です", "今日は晴れです")
        assert 0.0 < f1 < 1.0

    def test_gen_is_prefix_of_ref(self):
        f1 = compute_char_f1("abcdef", "abc")
        assert f1 > 0.0

    def test_ref_is_prefix_of_gen(self):
        f1 = compute_char_f1("abc", "abcdef")
        assert f1 > 0.0


class TestCheckJsonValidity:
    def test_valid_json(self):
        is_valid, has_keys = check_json_validity('{"a": 1}', ["a"])
        assert is_valid is True
        assert has_keys is True

    def test_valid_json_missing_keys(self):
        is_valid, has_keys = check_json_validity('{"a": 1}', ["a", "b"])
        assert is_valid is True
        assert has_keys is False

    def test_invalid_json(self):
        is_valid, has_keys = check_json_validity("not json", [])
        assert is_valid is False
        assert has_keys is False

    def test_json_in_code_block(self):
        is_valid, has_keys = check_json_validity('```json\n{"a": 1}\n```', ["a"])
        assert is_valid is True
        assert has_keys is True

    def test_json_in_plain_code_block(self):
        is_valid, has_keys = check_json_validity('```\n{"a": 1}\n```', ["a"])
        assert is_valid is True
        assert has_keys is True

    def test_embedded_json(self):
        is_valid, has_keys = check_json_validity('Here is the result: {"x": 42} end', ["x"])
        assert is_valid is True
        assert has_keys is True

    def test_empty_keys_no_check(self):
        is_valid, has_keys = check_json_validity('{"a": 1}', [])
        assert is_valid is True
        assert has_keys is True

    def test_non_dict_json(self):
        is_valid, has_keys = check_json_validity("[1, 2, 3]", [])
        assert is_valid is True


class TestLoadJsonl:
    def test_load_valid_jsonl(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n')
        records = load_jsonl(path)
        assert len(records) == 2
        assert records[0] == {"a": 1}
        assert records[1] == {"b": 2}

    def test_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_jsonl(path) == []

    def test_blank_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"a": 1}\n\n\n{"b": 2}\n')
        records = load_jsonl(path)
        assert len(records) == 2

    def test_unicode_content(self, tmp_path: Path):
        path = tmp_path / "unicode.jsonl"
        path.write_text('{"text": "日本語テスト"}\n', encoding="utf-8")
        records = load_jsonl(path)
        assert records[0]["text"] == "日本語テスト"

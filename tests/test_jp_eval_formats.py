"""Tests for Japanese eval task formatters (scripts/jp_eval_formats.py)."""
from __future__ import annotations

from scripts.jp_eval_formats import format_jcommonsenseqa, format_jnli, format_jsquad


class TestFormatJCommonsenseQA:
    def test_returns_prompt_and_expected(self):
        item = {
            "question": "日本の首都は？",
            "choice0": "東京",
            "choice1": "大阪",
            "choice2": "京都",
            "choice3": "名古屋",
            "choice4": "福岡",
            "label": 0,
        }
        prompt, expected = format_jcommonsenseqa(item)
        assert "日本の首都は？" in prompt
        assert "0: 東京" in prompt
        assert expected == "0"

    def test_label_as_string(self):
        item = {
            "question": "test",
            "choice0": "a",
            "choice1": "b",
            "choice2": "c",
            "choice3": "d",
            "choice4": "e",
            "label": 3,
        }
        _, expected = format_jcommonsenseqa(item)
        assert expected == "3"


class TestFormatJNLI:
    def test_entailment_string_label(self):
        item = {
            "premise": "猫がいる",
            "hypothesis": "動物がいる",
            "label": "entailment",
        }
        prompt, expected = format_jnli(item)
        assert "猫がいる" in prompt
        assert expected == "含意"

    def test_contradiction_int_label(self):
        item = {
            "premise": "テスト",
            "hypothesis": "テスト2",
            "label": 1,
        }
        _, expected = format_jnli(item)
        assert expected == "矛盾"

    def test_neutral_int_label(self):
        item = {
            "premise": "テスト",
            "hypothesis": "テスト2",
            "label": 2,
        }
        _, expected = format_jnli(item)
        assert expected == "中立"

    def test_sentence1_fallback(self):
        item = {
            "sentence1": "前提文",
            "sentence2": "仮説文",
            "label": 0,
        }
        prompt, _ = format_jnli(item)
        assert "前提文" in prompt

    def test_unknown_label_defaults_neutral(self):
        item = {
            "premise": "x",
            "hypothesis": "y",
            "label": "unknown",
        }
        _, expected = format_jnli(item)
        assert expected == "中立"


class TestFormatJSQuAD:
    def test_dict_answers(self):
        item = {
            "context": "東京は日本の首都です。",
            "question": "日本の首都は？",
            "answers": {"text": ["東京"]},
        }
        prompt, expected = format_jsquad(item)
        assert "日本の首都は？" in prompt
        assert expected == "東京"

    def test_empty_answers(self):
        item = {
            "context": "テスト",
            "question": "何？",
            "answers": {},
        }
        _, expected = format_jsquad(item)
        assert expected == ""

    def test_list_answers(self):
        item = {
            "context": "テスト",
            "question": "何？",
            "answers": [{"text": "答え"}],
        }
        _, expected = format_jsquad(item)
        assert expected == "答え"

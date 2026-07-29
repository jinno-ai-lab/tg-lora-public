"""First-pass coverage for the JSON-extraction quality scorer.

``src/eval/eval_json_extraction`` is the TG-LoRA experiment's **headline quality
metric** (perplexity's replacement — does the model emit *correct* structured
output?). It is wired into the base-model go/no-go gate
(``scripts/eval_base_model_json.py``), both generation-eval wrappers
(``src/eval/json_generation.py`` / ``jsonex_generation.py``), and documented as
the headline metric in ``src/training/config_schema.py``. It had **zero** direct
tests, so this file pins its load-bearing contract and the one observed defect.

The observed failure this file was written to lock: a perfect JSON object
followed by a short explanatory note that happens to contain a ``}`` was scored
as **completely invalid** (``valid=0``). Root cause: the lenient extractor used
a greedy ``\\{.*\\}`` regex, which spans from the first ``{`` to the *last* ``}``
in the text — capturing trailing prose and making ``json.loads`` fail. Models
routinely append a one-line explanation after JSON, so this is a realistic
false-negative on the headline metric, not a contrived adversarial input.
"""

from __future__ import annotations

from src.eval.eval_json_extraction import (
    extract_json,
    score_json_extraction,
    score_single,
)

GOLD_MEETING = {
    "type": "meeting",
    "attendee": "中村太郎",
    "date": "2026-12-09",
    "start": "12:15",
    "end": "14:45",
    "location": "タワービル12F",
    "priority": "high",
    "duration_minutes": 150,
}
GOLD_TX = {
    "type": "transaction",
    "item": "ノートPC",
    "quantity": 24,
    "unit_price": 18000,
    "total_cost": 432000,
    "counterparty": "株式会社A",
}
GOLD_PERSON = {
    "type": "person",
    "name": "中村太郎",
    "role": "CTO",
    "department": "開発部",
    "contact": "taro.1@acme.co.jp",
}

_PERFECT_MEETING_JSON = (
    '{"type":"meeting","attendee":"中村太郎","date":"2026-12-09",'
    '"start":"12:15","end":"14:45","location":"タワービル12F",'
    '"priority":"high","duration_minutes":150}'
)


class TestExtractJson:
    def test_strict_whole_text_parses_directly(self):
        obj, was_strict = extract_json(_PERFECT_MEETING_JSON)
        assert obj == GOLD_MEETING
        assert was_strict is True

    def test_markdown_fenced_is_valid_but_not_strict(self):
        fenced = f"```json\n{_PERFECT_MEETING_JSON}\n```"
        obj, was_strict = extract_json(fenced)
        assert obj == GOLD_MEETING
        # Surrounding fences/prose → lenient path → not strict.
        assert was_strict is False

    def test_trailing_assistant_token_is_stripped(self):
        for tok in ("<|im_end|>", "</s>", "<|eot_id|>"):
            obj, _ = extract_json(_PERFECT_MEETING_JSON + tok)
            assert obj == GOLD_MEETING, f"token {tok!r} broke strict parse"

    def test_no_json_returns_none(self):
        assert extract_json("会議の情報です。構造化出力なし。")[0] is None

    def test_trailing_prose_with_brace_does_not_invalidate_json(self):
        # OBSERVED FAILURE (was RED before the fix): a perfect JSON object
        # followed by an explanatory note containing a ``}`` was scored valid=0.
        # Models commonly append a one-line note after JSON; the greedy
        # ``\{.*\}`` spanned into the prose and broke json.loads.
        text = _PERFECT_MEETING_JSON + "\n(注意: { } は区切り文字です)"
        obj, was_strict = extract_json(text)
        assert obj == GOLD_MEETING, "perfect JSON + brace-prose must still parse"
        assert was_strict is False  # prose present → not a strict whole-text parse

    def test_braces_inside_string_values_are_ignored(self):
        # A ``}`` that appears *inside* a JSON string value must not be treated
        # as the object terminator.
        obj, _ = extract_json('{"type":"person","name":"a}b","role":"x","department":"d","contact":"c"}')
        assert obj is not None
        assert obj["name"] == "a}b"


class TestScoreSingle:
    def test_perfect_meeting_scores_all_ones(self):
        s = score_single(_PERFECT_MEETING_JSON, GOLD_MEETING)
        for key in ("valid", "strict_valid", "type_correct", "field_f1",
                    "exact_match", "computed_accuracy", "combined"):
            assert s[key] == 1.0, f"{key}={s[key]}"

    def test_wrong_computed_field_keeps_valid_but_zeroes_computed(self):
        # The graded signal: arithmetic wrong → computed_accuracy=0, the
        # surrounding 7/8 fields still count toward field_f1.
        wrong = _PERFECT_MEETING_JSON.replace('"duration_minutes":150', '"duration_minutes":140')
        s = score_single(wrong, GOLD_MEETING)
        assert s["valid"] == 1.0
        assert s["computed_accuracy"] == 0.0
        assert s["field_f1"] == 7 / 8
        assert s["exact_match"] == 0.0

    def test_thousands_separated_string_total_matches_numeric_gold(self):
        # A formatting quirk ("432,000") must not be scored as a math error.
        pred = (
            '{"type":"transaction","item":"ノートPC","quantity":24,'
            '"unit_price":18000,"total_cost":"432,000","counterparty":"株式会社A"}'
        )
        s = score_single(pred, GOLD_TX)
        assert s["computed_accuracy"] == 1.0
        assert s["exact_match"] == 1.0

    def test_off_by_one_total_does_not_match(self):
        pred = (
            '{"type":"transaction","item":"ノートPC","quantity":24,'
            '"unit_price":18000,"total_cost":432001,"counterparty":"株式会社A"}'
        )
        s = score_single(pred, GOLD_TX)
        assert s["computed_accuracy"] == 0.0
        assert s["exact_match"] == 0.0

    def test_person_has_no_computed_field(self):
        pred = (
            '{"type":"person","name":"中村太郎","role":"CTO",'
            '"department":"開発部","contact":"taro.1@acme.co.jp"}'
        )
        s = score_single(pred, GOLD_PERSON)
        assert s["computed_accuracy"] is None
        assert s["exact_match"] == 1.0


class TestScoreJsonExtraction:
    def test_computed_accuracy_excludes_person_records(self):
        # person records (computed_accuracy=None) must not dilute the metric;
        # computed_accuracy is averaged ONLY over meeting/transaction records.
        person_pred = (
            '{"type":"person","name":"中村太郎","role":"CTO",'
            '"department":"開発部","contact":"taro.1@acme.co.jp"}'
        )
        # A transaction that gets total_cost WRONG.
        tx_wrong = (
            '{"type":"transaction","item":"ノートPC","quantity":24,'
            '"unit_price":18000,"total_cost":999,"counterparty":"株式会社A"}'
        )
        scores = score_json_extraction([person_pred, tx_wrong], [GOLD_PERSON, GOLD_TX])
        # Only the transaction contributes → computed_accuracy = 0.0, NOT 0.5.
        assert scores["computed_accuracy"] == 0.0

    def test_empty_input_returns_empty_dict(self):
        assert score_json_extraction([], []) == {}

    def test_length_mismatch_raises(self):
        import pytest
        with pytest.raises(AssertionError):
            score_json_extraction(["a"], [])

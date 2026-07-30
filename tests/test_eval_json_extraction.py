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

    def test_escaped_quote_inside_string_value_is_not_a_terminator(self):
        # An *escaped* quote (``\"``) inside a JSON string value must not close
        # the string for the balanced-brace scanner. The escape-tracking branch
        # of ``_first_json_object`` (the ``\\`` set-escape / clear-escape pair)
        # is only LOAD-BEARING when a brace follows the escaped quote: if the
        # escaped ``"`` wrongly closed the string, that trailing ``}`` would be
        # counted as the object terminator, prematurely ending the span and
        # failing to parse. So the value below carries ``\"}`` (escaped quote
        # immediately followed by a brace). Trailing prose forces the lenient
        # scanner (the strict whole-text parse fails on the trailer).
        text = (
            r'{"type":"person","name":"a\"}b","role":"x",'
            r'"department":"d","contact":"c"} (note)'
        )
        obj, was_strict = extract_json(text)
        assert obj is not None, "escaped quote must not break extraction"
        assert obj["name"] == 'a"}b'
        assert was_strict is False  # trailing prose → lenient path

    def test_invalid_first_balanced_span_recovers_to_next_object(self):
        # The lenient scanner must not give up when the *first* brace-balanced
        # span fails to parse — it must reset and try the next candidate. A
        # model emitting a garbled fragment before the real object is the
        # realistic trigger; a naive scanner that returns on the first balanced
        # span would return None here (false-negative on the headline metric).
        text = '{ not json } {"type":"person","name":"x","role":"r","department":"d","contact":"c"}'
        obj, was_strict = extract_json(text)
        assert obj is not None, "scanner must recover past an invalid first span"
        assert obj["type"] == "person"
        assert obj["name"] == "x"
        assert was_strict is False


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

    def test_non_numeric_string_for_computed_field_scores_zero_not_crash(self):
        # A non-numeric placeholder ("N/A", "TBD") for a COMPUTED (numeric) gold
        # field must score computed_accuracy=0 — not crash the scorer. This is
        # the ``_field_match`` numeric-conversion error branch (``float(pred)``
        # raises on an unparseable string): without the try/except, a single
        # such prediction would take down the whole eval pass with an uncaught
        # ValueError.
        pred = (
            '{"type":"transaction","item":"ノートPC","quantity":24,'
            '"unit_price":18000,"total_cost":"N/A","counterparty":"株式会社A"}'
        )
        s = score_single(pred, GOLD_TX)
        assert s["valid"] == 1.0           # still valid JSON
        assert s["computed_accuracy"] == 0.0  # total_cost not a number → wrong
        assert s["exact_match"] == 0.0

    def test_person_has_no_computed_field(self):
        pred = (
            '{"type":"person","name":"中村太郎","role":"CTO",'
            '"department":"開発部","contact":"taro.1@acme.co.jp"}'
        )
        s = score_single(pred, GOLD_PERSON)
        assert s["computed_accuracy"] is None
        assert s["exact_match"] == 1.0

    def test_non_dict_gold_does_not_crash_scorer(self):
        # A gold completion that parsed as valid JSON but is NOT a dict — an
        # array (``[1,2,3]``), scalar (``42``), bool, null, or bare string from
        # a mis-generated or hand-edited dataset line — must not crash the
        # scorer. The prediction side is already guarded (extract_json returns
        # dict|None), but the gold side assumed a dict: ``gold.get("type")`` /
        # ``set(gold.keys())`` raised an uncaught AttributeError that aborted
        # the whole eval pass — the same "valid JSON, non-dict, dict-only
        # access" defect class as eval_format (1d6e7d4) and run_metrics
        # (a5506f7). A non-dict gold scores only the prediction's validity;
        # type/field/exact stay 0 and computed_accuracy stays None.
        pred = _PERFECT_MEETING_JSON
        for bad_gold in ([1, 2, 3], 42, True, None, "just a string"):
            s = score_single(pred, bad_gold)
            assert s["valid"] == 1.0, f"gold={bad_gold!r}: prediction still parses"
            assert s["strict_valid"] == 1.0
            assert s["type_correct"] == 0.0   # a non-dict gold has no type
            assert s["field_f1"] == 0.0
            assert s["exact_match"] == 0.0
            assert s["computed_accuracy"] is None  # excluded from the mean
            assert s["combined"] == 0.3           # 0.3*valid + 0.2*0 + 0.5*0
        # The aggregate path — the real entry point the wrappers
        # (json_generation / jsonex_generation) call with golds read straight
        # out of json.loads(record["completion"]) — must not crash either.
        scores = score_json_extraction([pred, pred], [[1, 2, 3], 42])
        assert scores["valid"] == 1.0
        assert scores["type_correct"] == 0.0
        assert scores["field_f1"] == 0.0
        # Neither gold had a computed field, so comp_n stays 0 → 0.0, not a crash.
        assert scores["computed_accuracy"] == 0.0


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

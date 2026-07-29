"""First-pass coverage for the two JSON-extraction **generation wrappers**.

``src/eval/eval_json_extraction`` (the pure scorer) got first coverage in
``test_eval_json_extraction.py``. This file covers the two torch-dependent
wrappers that *drive* it — the layer that turns a model into scored headline
metrics. Both are LIVE training-loop wiring with **zero** direct tests:

* ``src/eval/json_generation.py`` — ``generate_predictions`` /
  ``generate_and_score_json`` (batched greedy generation + left-padding).
  Feeds the base-model go/no-go difficulty gate
  (``scripts/eval_base_model_json.py``) and the training-loop JSON
  generation-quality eval (``train_tg_lora.py``, the ``json_eval_records`` hook,
  also measured under LAWA shadow weights).
* ``src/eval/jsonex_generation.py`` — ``generate_json_completions`` /
  ``evaluate_json_extraction_run`` (per-record generation). Feeds the §5.2
  gold-eval hook (``train_tg_lora.py``, the ``gold_eval_records`` hook) whose
  gold>=G* stop the analyzer resolves post-hoc.

The load-bearing contract this file pins: **a model that emits the gold
completion must score perfectly** (valid=strict_valid=field_f1=exact_match=
computed_accuracy=combined=1.0) end-to-end through generation → prompt
reconstruction → completion slicing → stop-token stripping → scoring. A
regression in any of those four steps would *silently corrupt the headline
quality metric* (perplexity's replacement) — exactly the silent-corruption
class the experiment cannot afford — yet currently no test would catch it.

These are realistic-contract tests over a gold-emitting model, NOT adversarial
corrupt-input synthesis: the model behaviour under test is the cleanest
possible output, so any sub-1.0 score is unambiguously a wrapper defect.
"""

from __future__ import annotations

import json

import torch

from src.eval.eval_json_extraction import score_json_extraction
from src.eval.json_generation import (
    _prompt_from_record,
    generate_and_score_json,
    generate_predictions,
)
from src.eval.jsonex_generation import (
    build_prompt_prefix,
    evaluate_json_extraction_run,
    generate_json_completions,
)

SYSTEM_INSTR = (
    "以下の文章を所定のJSONスキーマに変換してください。"
    "不要な情報は無視し、正規化して出力すること。"
)


def _record(prompt: str, gold: dict) -> dict:
    """Build a ChatML record in the exact shape ``generate_json_extraction_data`` emits."""
    completion = json.dumps(gold, ensure_ascii=False)
    text = (
        f"<|im_start|>user\n{SYSTEM_INSTR}\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )
    return {"text": text, "prompt": prompt, "completion": completion, "category": gold["type"]}


def _sample_records() -> list[dict]:
    meeting = {
        "type": "meeting",
        "attendee": "中村太郎",
        "date": "2026-12-09",
        "start": "12:15",
        "end": "14:45",
        "location": "タワービル12F",
        "priority": "high",
        "duration_minutes": 150,
    }
    person = {
        "type": "person",
        "name": "鈴木花子",
        "role": "CTO",
        "department": "開発部",
        "contact": "hanako.1@acme.co.jp",
    }
    tx = {
        "type": "transaction",
        "item": "ノートPC",
        "quantity": 24,
        "unit_price": 18000,
        "total_cost": 432000,
        "counterparty": "株式会社Aテック",
    }
    return [
        _record("12/9の12時15分〜14時45分、タワービル12Fで中村太郎との会議。優先度高。", meeting),
        _record("鈴木花子は開発部のCTO。連絡先はhanako.1@acme.co.jp。", person),
        _record("ノートPCを24個、単価18000円で株式会社Aテックに発注。", tx),
        # A second meeting of DIFFERENT length exercises left-padding slicing
        # under varied prompt widths inside one batch.
        _record("会議: 高橋健一, 2026-04-23 09:00-09:45, Room B, 優先度 medium", {
            "type": "meeting", "attendee": "高橋健一", "date": "2026-04-23",
            "start": "09:00", "end": "09:45", "location": "Room B",
            "priority": "medium", "duration_minutes": 45,
        }),
    ]


# --------------------------------------------------------------------------- #
# Faithful char-level mock tokenizer + gold-emitting mock model.               #
# --------------------------------------------------------------------------- #


class _TokenizedOutput(dict):
    """Dict subclass with ``.to(device)`` (mirrors HF tokenizer output)."""

    def to(self, device):  # noqa: D401, ANN001 - match HF ergonomics
        return self


class _CharTokenizer:
    """Round-trips text <-> ids one char per id (id 0 reserved for padding).

    Supports the call/decode/batch_decode surface and the padding_side /
    pad_token / eos_token attributes the wrappers touch, so the generation
    path (left-padding, new-token slicing, batch_decode) is exercised for real.
    """

    padding_side = "right"
    pad_token = None
    eos_token = "<|im_end|>"
    eos_token_id = 0
    pad_token_id = 0

    def _encode(self, text: str) -> list[int]:
        return [ord(c) + 1 for c in text]

    def _decode_ids(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(i - 1) for i in ids if i > 0)  # id 0 (pad) -> skipped

    def __call__(self, text, return_tensors=None, padding=False, **kwargs):
        if isinstance(text, str):
            text = [text]
        seqs = [self._encode(t) for t in text]
        if padding:
            width = max(len(s) for s in seqs)
            if self.padding_side == "left":
                seqs = [[0] * (width - len(s)) + s for s in seqs]
            else:
                seqs = [s + [0] * (width - len(s)) for s in seqs]
        ids = torch.tensor(seqs, dtype=torch.long)
        return _TokenizedOutput(input_ids=ids, attention_mask=(ids != 0).long())

    def decode(self, ids, skip_special_tokens=False):
        return self._decode_ids(ids)

    def batch_decode(self, batch, skip_special_tokens=False):
        if hasattr(batch, "tolist"):
            batch = batch.tolist()
        return [self._decode_ids(row) for row in batch]


class _GoldEmittingGenerate:
    """``model.generate`` stand-in: emits each prompt's gold completion + stop.

    Decodes each (left-padded) input row back to its prompt text, looks up the
    gold completion, and returns ``input_ids`` concatenated with a per-row
    ``[gold tokens, <|im_end|>]`` block right-padded to a common width. This
    drives both the batched path (multiple rows, varied lengths) and the
    per-record path identically.
    """

    def __init__(self, tokenizer: _CharTokenizer, prompt_to_gold: dict[str, str]):
        self._tok = tokenizer
        self._p2g = prompt_to_gold

    def __call__(self, **kwargs):
        ids = kwargs["input_ids"]
        rows = [self._tok._decode_ids(row) for row in ids]
        golds = [self._tok._encode(self._p2g[row] + "<|im_end|>") for row in rows]
        width = max(len(g) for g in golds)
        block = torch.zeros(ids.shape[0], width, dtype=torch.long)
        for i, g in enumerate(golds):
            block[i, : len(g)] = torch.tensor(g)
        return torch.cat([ids, block], dim=1)


class _MockModel:
    training = False

    def parameters(self):
        return iter([torch.zeros(1, requires_grad=True)])

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


def _gold_emit_model(records: list[dict]) -> tuple[_MockModel, _CharTokenizer]:
    tok = _CharTokenizer()
    p2g = {_prompt_from_record(r): r["completion"] for r in records}
    model = _MockModel()
    model.generate = _GoldEmittingGenerate(tok, p2g)
    return model, tok


_PERFECT_KEYS = ("valid", "strict_valid", "type_correct", "field_f1",
                 "exact_match", "computed_accuracy", "combined")


# --------------------------------------------------------------------------- #
# End-to-end: gold-emitting model scores perfectly through both wrappers.     #
# --------------------------------------------------------------------------- #


class TestGoldEmittingModelScoresPerfect:
    def test_batched_generate_and_score_json(self):
        records = _sample_records()
        model, tok = _gold_emit_model(records)
        scores = generate_and_score_json(
            model, tok, records, batch_size=2, max_new_tokens=96, device="cpu"
        )
        for key in _PERFECT_KEYS:
            assert scores[key] == 1.0, f"gold-emitting model must score {key}=1.0, got {scores[key]}"

    def test_per_record_evaluate_json_extraction_run(self):
        records = _sample_records()
        model, tok = _gold_emit_model(records)
        scores = evaluate_json_extraction_run(
            model, tok, records, max_new_tokens=96, device="cpu"
        )
        for key in _PERFECT_KEYS:
            assert scores[key] == 1.0, f"gold-emitting model must score {key}=1.0, got {scores[key]}"

    def test_batched_generate_predictions_directly(self):
        # Drive the batched generator with batch_size that does not divide the
        # record count, exercising the partial final batch + left-padding slice.
        records = _sample_records()  # 4 records
        model, tok = _gold_emit_model(records)
        preds = generate_predictions(
            model, tok, records, batch_size=3, max_new_tokens=96, device="cpu"
        )
        golds = [json.loads(r["completion"]) for r in records]
        scores = score_json_extraction(preds, golds)
        for key in _PERFECT_KEYS:
            assert scores[key] == 1.0, f"batch_size=3 (partial batch) {key}={scores[key]}"


# --------------------------------------------------------------------------- #
# Cross-hook comparability: both hooks must feed the SAME prompt to the model. #
# --------------------------------------------------------------------------- #


class TestPromptReconstructionConsistency:
    def test_both_hooks_reconstruct_identical_prompt(self):
        # The two headline-metric hooks (§5.2 per-record gold eval vs the
        # batched json-eval/go-no-go gate) must reconstruct the SAME generation
        # prompt from a record. A divergence would make the two hooks score the
        # same model+data differently — silently making the §5.2 gold>=G* stop
        # incomparable against the json-eval curve. Pin they agree on real data.
        records = _sample_records()
        for r in records:
            assert _prompt_from_record(r) == build_prompt_prefix(r), (
                "generation hooks diverged on prompt reconstruction"
            )

    def test_prompt_reconstruction_excludes_gold_completion(self):
        # The reconstructed prompt must END at the assistant header and contain
        # NONE of the gold completion (else the model would be shown its answer
        # and the metric would be meaningless).
        records = _sample_records()
        for r in records:
            prompt = _prompt_from_record(r)
            assert prompt.endswith("<|im_start|>assistant\n")
            assert r["completion"] not in prompt


class TestPredictionHygiene:
    def test_predictions_strip_trailing_stop_token(self):
        records = _sample_records()
        model, tok = _gold_emit_model(records)
        for pred in generate_predictions(model, tok, records, batch_size=2, device="cpu"):
            assert "<|im_end|>" not in pred
            assert not pred.endswith("<|endoftext|>")

    def test_predictions_strip_trailing_stop_token_per_record(self):
        records = _sample_records()
        model, tok = _gold_emit_model(records)
        preds, _ = generate_json_completions(model, tok, records, device="cpu")
        for pred in preds:
            assert "<|im_end|>" not in pred


class TestPreviewAndEmpty:
    def test_preview_omitted_by_default(self):
        records = _sample_records()
        model, tok = _gold_emit_model(records)
        scores = generate_and_score_json(model, tok, records, device="cpu")
        assert "_preview" not in scores

    def test_preview_present_when_requested(self):
        records = _sample_records()
        model, tok = _gold_emit_model(records)
        scores = generate_and_score_json(model, tok, records, n_preview=2, device="cpu")
        preview = scores["_preview"]
        assert len(preview) == 2
        assert {p["prompt"] for p in preview} == {r["prompt"] for r in records[:2]}
        for item in preview:
            assert set(item) == {"prompt", "gold", "pred"}

    def test_empty_records_return_empty_dict_batched(self):
        model, tok = _gold_emit_model(_sample_records())
        assert generate_and_score_json(model, tok, [], device="cpu") == {}

    def test_empty_records_return_empty_dict_per_record(self):
        model, tok = _gold_emit_model(_sample_records())
        assert evaluate_json_extraction_run(model, tok, [], device="cpu") == {}


class TestStateRestoration:
    def test_batched_restores_padding_side_and_model_mode(self):
        records = _sample_records()
        model, tok = _gold_emit_model(records)
        model.train()
        assert tok.padding_side == "right"
        generate_predictions(model, tok, records, batch_size=2, device="cpu")
        # The batched path temporarily sets left-padding + eval mode.
        assert tok.padding_side == "right"
        assert model.training is True

    def test_per_record_restores_model_mode(self):
        records = _sample_records()
        model, tok = _gold_emit_model(records)
        model.train()
        generate_json_completions(model, tok, records, device="cpu")
        assert model.training is True

import json
from pathlib import Path
import pytest

from scripts.eval_downstream import (
    compute_char_f1,
    check_json_validity,
    is_char_f1_degenerate_input,
    aggregate_char_f1,
)


def test_compute_char_f1():
    # Identical strings
    assert compute_char_f1("こんにちは", "こんにちは") == 1.0

    # Empty strings
    assert compute_char_f1("", "こんにちは") == 0.0
    assert compute_char_f1("こんにちは", "") == 0.0
    assert compute_char_f1("", "") == 0.0

    # Whitespace stripping
    assert compute_char_f1("こん にち は", "こんにちは") == 1.0

    # Partial match
    # Match length = 3 ("にちは"), precision = 3/3 = 1.0, recall = 3/5 = 0.6
    # F1 = 2 * (1.0 * 0.6) / (1.0 + 0.6) = 1.2 / 1.6 = 0.75
    assert compute_char_f1("こんにちは", "にちは") == pytest.approx(0.75)


def test_is_char_f1_degenerate_input_flags_empty_sides():
    # Degenerate = empty reference or empty generation -> F1 is *undefined*, not
    # genuinely zero (these are the items that must be surfaced, not averaged in).
    assert is_char_f1_degenerate_input("", "こんにちは") is True
    assert is_char_f1_degenerate_input("こんにちは", "") is True
    assert is_char_f1_degenerate_input("", "") is True
    # Whitespace-only is degenerate too (collapses to empty after cleaning).
    assert is_char_f1_degenerate_input("   ", "\n\t") is True
    # A genuine comparison is NOT degenerate, even a total non-match.
    assert is_char_f1_degenerate_input("abc", "xyz") is False


def test_aggregate_char_f1_surfaces_degenerate_not_silent():
    # One genuine match (F1=1.0) + one empty-generation collapse (degenerate).
    # OLD behavior averaged the silent 0.0 into the mean -> 0.5, hiding the collapse.
    per_item = [
        {"char_f1": 1.0, "degenerate": False},
        {"char_f1": 0.0, "degenerate": True},
    ]
    summary = aggregate_char_f1(per_item)
    # The headline mean runs over the comparable item only (d9ca7f5 survivor
    # discipline), so the degenerate item cannot drag it to 0.5.
    assert summary["mean_char_f1"] == pytest.approx(1.0)
    assert summary["comparable"] == 1
    assert summary["degenerate_inputs"] == 1
    assert summary["total"] == 2
    assert summary["incomplete"] is True


def test_aggregate_char_f1_all_degenerate_surfaces_total_collapse():
    # If every item is degenerate the headline must not masquerade as a clean mean.
    per_item = [
        {"char_f1": 0.0, "degenerate": True},
        {"char_f1": 0.0, "degenerate": True},
    ]
    summary = aggregate_char_f1(per_item)
    assert summary["comparable"] == 0
    assert summary["degenerate_inputs"] == 2
    assert summary["total"] == 2
    assert summary["incomplete"] is True
    # No comparable items -> mean is 0.0, but `incomplete` is the honest signal.
    assert summary["mean_char_f1"] == 0.0


def test_aggregate_char_f1_clean_corpus_no_degenerate():
    per_item = [
        {"char_f1": 0.8, "degenerate": False},
        {"char_f1": 0.6, "degenerate": False},
    ]
    summary = aggregate_char_f1(per_item)
    assert summary["mean_char_f1"] == pytest.approx(0.7)
    assert summary["comparable"] == 2
    assert summary["degenerate_inputs"] == 0
    assert summary["total"] == 2
    assert summary["incomplete"] is False


def test_aggregate_char_f1_empty_corpus():
    summary = aggregate_char_f1([])
    assert summary["total"] == 0
    assert summary["comparable"] == 0
    assert summary["degenerate_inputs"] == 0
    assert summary["incomplete"] is False
    assert summary["mean_char_f1"] == 0.0


def test_check_json_validity():
    # Valid JSON dict with all keys
    gen_ok = '{"name": "Yamada", "age": 25}'
    is_valid, has_keys = check_json_validity(gen_ok, ["name", "age"])
    assert is_valid is True
    assert has_keys is True

    # Valid JSON dict with missing keys
    is_valid, has_keys = check_json_validity(gen_ok, ["name", "gender"])
    assert is_valid is True
    assert has_keys is False

    # Wrappers: markdown code blocks
    gen_markdown = '```json\n{"name": "Yamada", "age": 25}\n```'
    is_valid, has_keys = check_json_validity(gen_markdown, ["name", "age"])
    assert is_valid is True
    assert has_keys is True

    # Fallback to inner braces
    gen_text_around = '結果は以下の通りです：\n{"name": "Yamada", "age": 25}\nご確認ください。'
    is_valid, has_keys = check_json_validity(gen_text_around, ["name", "age"])
    assert is_valid is True
    assert has_keys is True

    # Invalid JSON
    gen_bad = '{"name": "Yamada", "age": 25'
    is_valid, has_keys = check_json_validity(gen_bad, ["name"])
    assert is_valid is False
    assert has_keys is False


def test_downstream_dataset_files():
    jp_path = Path("data/downstream/jp_capability.jsonl")
    json_path = Path("data/downstream/format_json.jsonl")
    
    # Auto-create dummy files if they do not exist in the test environment
    if not jp_path.parent.exists():
        jp_path.parent.mkdir(parents=True, exist_ok=True)
    if not jp_path.exists():
        with open(jp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "Hello", "completion": "こんにちは"}) + "\n")
    if not json_path.exists():
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "Format", "completion": "{}"}) + "\n")
            
    assert jp_path.exists(), "jp_capability.jsonl dataset is missing"
    assert json_path.exists(), "format_json.jsonl dataset is missing"
    
    # Validate structure
    for path in [jp_path, json_path]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                assert "prompt" in data
                assert "completion" in data
                assert len(data["prompt"]) > 0
                assert len(data["completion"]) > 0

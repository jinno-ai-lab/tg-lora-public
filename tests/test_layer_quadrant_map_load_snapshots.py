"""Behavior tests for ``scripts.layer_quadrant_map.load_snapshots``: a stray
``.pt`` that isn't a cycle-suffixed trajectory-delta snapshot is SKIPPED, not
crashed on.

Crash class: the artifacts dir an operator points ``--art-dir`` at can contain a
``.pt`` that isn't a trajectory-delta snapshot — ``training_state.pt`` /
``optimizer.pt`` / ``adapter_model.pt`` / a bare saved tensor. Pre-fix,
``load_snapshots`` did ``cyc = int(name.split("_")[-1])`` and then
``t["delta_tensors"]`` with NO guard, so ONE such unrelated file aborted the
entire analysis with an uncaught ``ValueError`` (``int("state")``) /
``KeyError`` / ``TypeError`` instead of degrading over the conforming files.

Post-fix the cycle is extracted via a ``_<digits>`` regex predicate (returns
``None`` on no match) and the loaded object is shape-checked, so non-conforming
files are skipped with a stderr warning. This is the same skip-over-crash idiom
as the hardened ``src.utils.checkpoint._sorted_trajectory_delta_artifact_files``
regex — the ad-hoc loader here predates and didn't share it. Distinct crash
class from the json/yaml non-dict-after-deserialize guards (this is an
int-parse-on-filename + key-access-on-torch.load result, not a dict access on a
text-deserializer result).
"""

from __future__ import annotations

import pytest
import torch

from scripts.layer_quadrant_map import _cycle_from_artifact_stem, load_snapshots


# ── the cycle predicate (mutation-provable in isolation) ───────────────


@pytest.mark.parametrize(
    "stem,expected",
    [
        # canonical producer names end in _<6-digit cycle>
        ("tg_lora_after_pilot_cycle_000005", 5),
        ("tg_lora_after_speculative_update_cycle_000011", 11),
        # any trailing _<digits> is accepted (preserves prior happy-path semantics)
        ("snapshot_cycle_42", 42),
        ("c_0", 0),
        # stray / non-conforming stems → None (skip, don't crash)
        ("training_state", None),
        ("optimizer", None),
        ("adapter_model", None),
        ("model_final", None),
        ("cycle", None),  # trailing token is non-numeric
        ("", None),  # empty stem
    ],
)
def test_cycle_from_artifact_stem(stem: str, expected: int | None) -> None:
    assert _cycle_from_artifact_stem(stem) == expected


# ── behavior: stray files are skipped, conforming snapshots still load ──


def _save(path, name: str, content) -> None:
    torch.save(content, path / name)


def test_load_snapshots_skips_stray_non_cycle_pt(tmp_path, capsys) -> None:
    """A stray ``training_state.pt`` (no trailing ``_<digits>``) is skipped and
    the conforming snapshot still loads. Pre-fix this raised
    ``ValueError: invalid literal for int() with base 10: 'state'``."""
    _save(tmp_path, "training_state.pt", {"unrelated": "data"})
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000003.pt",
        {"delta_tensors": {"layer.0.lora_A.weight": torch.ones(2, 2)}},
    )

    snapshots = load_snapshots(str(tmp_path))

    assert sorted(snapshots.keys()) == [3]
    err = capsys.readouterr().err
    assert "training_state.pt" in err
    assert "skip" in err


def test_load_snapshots_skips_pt_missing_delta_tensors(tmp_path, capsys) -> None:
    """A cycle-suffixed ``.pt`` whose content lacks a ``delta_tensors`` mapping
    is skipped instead of ``KeyError``/``TypeError``."""
    _save(tmp_path, "state_000007.pt", {"optimizer_state": {}})  # right name, wrong shape
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000007.pt",
        {"delta_tensors": {"layer.0.lora_A.weight": torch.ones(2, 2)}},
    )

    snapshots = load_snapshots(str(tmp_path))

    assert sorted(snapshots.keys()) == [7]
    err = capsys.readouterr().err
    assert "state_000007.pt" in err
    assert "skip" in err


def test_load_snapshots_skips_bare_tensor_pt(tmp_path) -> None:
    """A ``.pt`` that loads to a bare tensor (not a dict) is skipped instead of
    ``TypeError: tensor indices must be...``."""
    _save(tmp_path, "junk_000009.pt", torch.zeros(3))
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000009.pt",
        {"delta_tensors": {"layer.0.lora_A.weight": torch.ones(2, 2)}},
    )

    snapshots = load_snapshots(str(tmp_path))

    assert sorted(snapshots.keys()) == [9]


def test_load_snapshots_happy_path_float_casts(tmp_path) -> None:
    """Regression guard: conforming snapshots still load, keyed by cycle, with
    values float-cast (the SVD path downstream needs floating-point tensors)."""
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000001.pt",
        {"delta_tensors": {"layer.0.lora_A.weight": torch.ones(2, 2, dtype=torch.int64)}},
    )
    _save(
        tmp_path,
        "tg_lora_after_speculative_update_cycle_000002.pt",
        {"delta_tensors": {"layer.1.lora_B.weight": torch.ones(2, 2)}},
    )

    snapshots = load_snapshots(str(tmp_path))

    assert sorted(snapshots.keys()) == [1, 2]
    assert snapshots[1]["layer.0.lora_A.weight"].dtype == torch.float32


def test_load_snapshots_all_stray_returns_empty(tmp_path, capsys) -> None:
    """A dir with only non-conforming files degrades to an empty result instead
    of crashing — the downstream ``main`` already errors on an empty cycle set,
    so an honest empty dict (not a traceback) is the correct failure shape."""
    _save(tmp_path, "training_state.pt", {"x": 1})
    _save(tmp_path, "optimizer.pt", [1, 2, 3])

    snapshots = load_snapshots(str(tmp_path))

    assert snapshots == {}
    err = capsys.readouterr().err
    assert "training_state.pt" in err and "optimizer.pt" in err

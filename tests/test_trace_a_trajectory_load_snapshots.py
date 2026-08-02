"""Behavior tests for ``scripts.trace_a_trajectory.load_trajectory_snapshots``:
a stray ``.pt`` that isn't a trajectory-delta snapshot is SKIPPED, not crashed on.

Crash class: the artifacts dir this script globs over can contain a ``.pt`` that
isn't a trajectory-delta snapshot — ``training_state.pt`` / ``optimizer.pt`` /
``adapter_model.pt`` / a bare saved tensor. Pre-fix, ``main`` did
``art = torch.load(f, ...)`` then ``art["delta_tensors"]`` unconditionally for
every ``*.pt`` (no cycle filter, no shape check), so ONE such unrelated file
aborted the entire trace with an uncaught ``KeyError`` (``'delta_tensors'``) /
``TypeError`` instead of degrading over the conforming snapshots.

Post-fix ``load_trajectory_snapshots`` shape-checks each loaded object (must be a
dict bearing a ``delta_tensors`` mapping) and skips non-conforming files with a
stderr warning. Same skip-over-crash idiom as
``scripts.layer_quadrant_map.load_snapshots`` (8b4372c) and the hardened
``src.utils.checkpoint._sorted_trajectory_delta_artifact_files`` regex. Distinct
crash class from the json/yaml non-dict-after-deserialize guards (this is a
key-access on a ``torch.load`` result over a ``*.pt`` glob, not a dict access on
a text-deserializer result).
"""

from __future__ import annotations

import torch

from scripts.trace_a_trajectory import load_trajectory_snapshots


def _save(path, name: str, content) -> None:
    torch.save(content, path / name)


def test_load_skips_stray_pt_missing_delta_tensors(tmp_path, capsys) -> None:
    """A stray ``training_state.pt`` (dict without ``delta_tensors``) is skipped
    and the conforming snapshot still loads. Pre-fix this raised
    ``KeyError: 'delta_tensors'``."""
    _save(tmp_path, "training_state.pt", {"unrelated": "data"})
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000003.pt",
        {"delta_tensors": {"layer.0.lora_A.weight": torch.ones(2, 2)}},
    )

    snapshots = load_trajectory_snapshots(tmp_path)

    assert len(snapshots) == 1
    assert "layer.0.lora_A.weight" in snapshots[0]
    err = capsys.readouterr().err
    assert "training_state.pt" in err
    assert "skip" in err


def test_load_skips_bare_tensor_pt(tmp_path) -> None:
    """A ``.pt`` that loads to a bare tensor (not a dict) is skipped instead of
    ``TypeError: tensor indices must be...``."""
    _save(tmp_path, "junk_000009.pt", torch.zeros(3))
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000009.pt",
        {"delta_tensors": {"layer.0.lora_A.weight": torch.ones(2, 2)}},
    )

    snapshots = load_trajectory_snapshots(tmp_path)

    assert len(snapshots) == 1


def test_load_skips_list_pt(tmp_path) -> None:
    """A ``.pt`` that loads to a list (optimizer-state-like) is skipped — the
    ``isinstance(art, dict)`` guard rejects it rather than crashing on
    ``art.get``."""
    _save(tmp_path, "optimizer_000004.pt", [{"momentum": 1}, {"momentum": 2}])
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000004.pt",
        {"delta_tensors": {"layer.0.lora_A.weight": torch.ones(2, 2)}},
    )

    snapshots = load_trajectory_snapshots(tmp_path)

    assert len(snapshots) == 1


def test_load_all_stray_returns_empty(tmp_path, capsys) -> None:
    """A dir with only non-conforming files degrades to an empty list instead of
    crashing — ``main`` errors loudly on an empty result, so an honest empty
    list (not a traceback) is the correct failure shape."""
    _save(tmp_path, "training_state.pt", {"x": 1})
    _save(tmp_path, "optimizer.pt", [1, 2, 3])
    _save(tmp_path, "adapter_model.pt", torch.zeros(2))

    snapshots = load_trajectory_snapshots(tmp_path)

    assert snapshots == []
    err = capsys.readouterr().err
    assert "training_state.pt" in err and "optimizer.pt" in err and "adapter_model.pt" in err


def test_load_preserves_sorted_filename_order(tmp_path) -> None:
    """Regression guard: snapshots load in sorted filename order — the per-cycle
    trajectory math downstream pairs ``all_deltas[c]`` with cycle ``c`` by
    position, so order is load-bearing, not cosmetic."""
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000002.pt",
        {"delta_tensors": {"m": torch.tensor([2.0])}},
    )
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000000.pt",
        {"delta_tensors": {"m": torch.tensor([0.0])}},
    )
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000005.pt",
        {"delta_tensors": {"m": torch.tensor([5.0])}},
    )

    snapshots = load_trajectory_snapshots(tmp_path)

    assert [s["m"].item() for s in snapshots] == [0.0, 2.0, 5.0]


def test_load_skips_dict_with_non_dict_delta_tensors(tmp_path, capsys) -> None:
    """A ``.pt`` whose ``delta_tensors`` value isn't a mapping (wrong shape) is
    skipped — ``isinstance(...get('delta_tensors'), dict)`` rejects it rather
    than letting the downstream ``.keys()``/``[tn]`` crash later."""
    _save(tmp_path, "weird_000001.pt", {"delta_tensors": [1, 2, 3]})
    _save(
        tmp_path,
        "tg_lora_after_pilot_cycle_000001.pt",
        {"delta_tensors": {"layer.0.lora_A.weight": torch.ones(2, 2)}},
    )

    snapshots = load_trajectory_snapshots(tmp_path)

    assert len(snapshots) == 1
    err = capsys.readouterr().err
    assert "weird_000001.pt" in err

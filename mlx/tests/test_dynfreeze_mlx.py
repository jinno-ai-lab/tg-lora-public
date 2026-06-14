"""Unit tests for the MLX DynamicFreezeController decision logic (adaptive τ).

Bypasses compute_r_A (needs a model) by injecting r_A history AND per-layer
peak directly, exercising the scale-invariant settled criterion + cap.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx.src.dynfreeze_mlx import DynamicFreezeController  # noqa: E402

LAYERS = list(range(24, 32))  # L24..L31


def _make(settled_ratio=0.15, window=5, stir=10, unfreeze_ratio=0.5,
          min_trainable=2) -> DynamicFreezeController:
    return DynamicFreezeController(
        settled_ratio=settled_ratio, window=window, stir_interval=stir,
        unfreeze_ratio=unfreeze_ratio, min_trainable=min_trainable,
        all_layer_indices=LAYERS,
    )


def _set_history(c: DynamicFreezeController, values: dict[int, float],
                 peak: float = 0.1) -> None:
    """Force each layer's r_A window to a fixed value, and set its peak.

    With settled_ratio=0.15 and peak=0.1 the threshold is 0.015, so
    values <= 0.015 are 'settled' and larger values are not.
    """
    c._r_A_history = {li: deque([values[li]] * c._window) for li in values}
    c._peak_r_A = {li: peak for li in values}


def test_freeze_capped_when_all_settled():
    c = _make(min_trainable=2)
    _set_history(c, {li: 0.001 for li in LAYERS})  # all settled
    out = c.decide_freeze(cycle=5)
    # cap: 8 layers, min_trainable=2 -> freeze at most 6 (L31..L26)
    assert out == [31, 30, 29, 28, 27, 26], out
    print("test_freeze_capped_when_all_settled: OK", out)


def test_freeze_no_cap_when_min_trainable_zero():
    c = _make(min_trainable=0)
    _set_history(c, {li: 0.001 for li in LAYERS})
    out = c.decide_freeze(cycle=5)
    assert out == [31, 30, 29, 28, 27, 26, 25, 24], out
    print("test_freeze_no_cap_when_min_trainable_zero: OK")


def test_freeze_stops_at_first_noisy():
    c = _make()
    hist = {li: 0.001 for li in LAYERS}
    hist[29] = 0.05  # not settled -> block stops before L29
    _set_history(c, hist)
    out = c.decide_freeze(cycle=5)
    assert out == [31, 30], out
    print("test_freeze_stops_at_first_noisy: OK", out)


def test_freeze_contiguous_extension():
    c = _make()
    c._frozen_block = [31, 30]
    c._peak_r_A = {li: 0.1 for li in LAYERS}
    hist = {li: 0.05 for li in LAYERS}
    hist[29] = 0.001  # only L29 newly settled, contiguous with block
    _set_history(c, hist)
    out = c.decide_freeze(cycle=5)
    assert out == [29], out
    print("test_freeze_contiguous_extension: OK", out)


def test_unfreeze_stir():
    c = _make(stir=10)
    c._frozen_block = [31, 30, 29]
    c._frozen_since_cycle = 5
    out = c.decide_unfreeze(cycle=16)  # frozen 11 cycles >= R=10
    assert out == [29], out
    print("test_unfreeze_stir: OK", out)


def test_unfreeze_upstream_reawaken():
    c = _make(unfreeze_ratio=0.5)
    c._frozen_block = [31, 30, 29]
    c._frozen_since_cycle = 100  # within stir interval
    # upstream neighbor L28 re-awakened to 0.06 > 0.5*peak(0.1)=0.05
    _set_history(c, {28: 0.06})
    out = c.decide_unfreeze(cycle=5)
    assert out == [29], out
    print("test_unfreeze_upstream_reawaken: OK", out)


def test_unfreeze_no_trigger():
    c = _make(stir=10, unfreeze_ratio=0.5)
    c._frozen_block = [31, 30, 29]
    c._frozen_since_cycle = 100
    _set_history(c, {28: 0.001})  # upstream quiet -> no re-awaken
    out = c.decide_unfreeze(cycle=102)  # frozen 2 cycles < 10
    assert out == [], out
    print("test_unfreeze_no_trigger: OK", out)


def test_settled_scale_invariance():
    """Same relative drop settles regardless of absolute r_A scale."""
    for scale in (0.1, 1.0, 10.0):  # different r_A magnitudes
        c = _make(settled_ratio=0.15)
        peak = 0.5 * scale
        c._peak_r_A = {31: peak}
        c._r_A_history = {31: deque([0.06 * scale] * 5)}  # 12% of peak -> settled
        assert c._is_quiet(31), f"scale {scale}: should be settled"
        c._r_A_history = {31: deque([0.4 * scale] * 5)}  # 80% of peak -> not
        assert not c._is_quiet(31), f"scale {scale}: should NOT be settled"
    print("test_settled_scale_invariance: OK (settles at same ratio across scales)")


def test_state_roundtrip():
    c = _make()
    c._frozen_block = [31, 30]
    c._frozen_since_cycle = 7
    c._r_A_history = {31: deque([0.1, 0.2], maxlen=5)}
    c._peak_r_A = {31: 0.9, 30: 0.8}
    s = c.state_dict()
    c2 = _make()
    c2.load_state_dict(s)
    assert c2.frozen_block == [31, 30]
    assert c2._frozen_since_cycle == 7
    assert c2._peak_r_A == {31: 0.9, 30: 0.8}
    print("test_state_roundtrip: OK (peak_r_A preserved)")


if __name__ == "__main__":
    test_freeze_capped_when_all_settled()
    test_freeze_no_cap_when_min_trainable_zero()
    test_freeze_stops_at_first_noisy()
    test_freeze_contiguous_extension()
    test_unfreeze_stir()
    test_unfreeze_upstream_reawaken()
    test_unfreeze_no_trigger()
    test_settled_scale_invariance()
    test_state_roundtrip()
    print("\nALL dynfreeze_mlx ADAPTIVE TESTS PASSED")

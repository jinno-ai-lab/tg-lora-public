"""Regression + behavior tests for ``scripts.analyze_trajectory_deltas``.

``compute_regime_inventory`` (GOAL §4 step 2 regime inventory) had a latent
``NameError``: its empty-``step_cosines`` early return referenced
``len(incregments)`` — an undefined name (typo for the ``increments``
parameter) — while the surrounding early returns (lines 123/145) correctly used
``increments``. A degenerate trajectory whose sampled tensors are all zero-norm
between consecutive steps makes every cosine ``n1 > 1e-10 and n2 > 1e-10`` guard
false, so ``step_cosines`` stays empty and execution reached the buggy line,
raising ``NameError`` instead of the documented ``{"n_steps": ...}`` return.

``ruff`` surfaced this as F821 (undefined name). This test feeds exactly that
degenerate input to lock the fix: post-fix the function returns the documented
dict; pre-fix it raised ``NameError``. The happy-path test documents the
stable-trajectory classification the function exists to produce.
"""

from __future__ import annotations

import pytest
import torch

from scripts.analyze_trajectory_deltas import compute_regime_inventory
from src.tg_lora.layer_type import LayerType


def test_compute_regime_inventory_zero_norm_increments_no_nameerror() -> None:
    """Empty-``step_cosines`` early return must not raise ``NameError``.

    The selected tensor is in a ``target_types`` type (so ``selected`` is
    non-empty and the line-122 guard passes), and every increment is zero so no
    cosine clears the ``n1 > 1e-10 and n2 > 1e-10`` guard — ``step_cosines`` is
    empty, the path that used to hit ``len(incregments)``.
    """
    name = "layer.0.attention_out"
    per_tensor = {name: {"layer_type": LayerType.ATTENTION_OUT.value}}
    zero = torch.zeros(8)
    increments = [{name: zero}, {name: zero}, {name: zero}]

    result = compute_regime_inventory(increments, per_tensor)

    assert result == {"n_steps": 3}


def test_compute_regime_inventory_stable_trajectory() -> None:
    """Identical non-zero increments ⇒ cosine 1.0 ⇒ fully stable inventory.

    Documents the happy path the regime inventory exists to classify: a
    trajectory that keeps a consistent direction grades as all-stable.
    """
    name = "block.2.deltanet"
    per_tensor = {name: {"layer_type": LayerType.DELTANET.value}}
    vec = torch.randn(16)
    increments = [{name: vec}, {name: vec}, {name: vec}]

    result = compute_regime_inventory(increments, per_tensor)

    assert result["n_steps"] == 3
    assert result["n_tensors_sampled"] == 1
    assert result["stable_fraction"] == pytest.approx(1.0)
    assert result["plateau_fraction"] == pytest.approx(0.0)
    assert result["transition_fraction"] == pytest.approx(0.0)
    assert result["mean_cosine"] == pytest.approx(1.0)

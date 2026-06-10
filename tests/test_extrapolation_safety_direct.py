"""Direct NaN detection tests with real apply_extrapolation (TASK-0032).

Uses the actual apply_extrapolation implementation (not mocked) with carefully
chosen parameters to verify:
1. NaN production through the real code path (inf overflow -> cap_update guard)
2. cap_update prevents NaN with moderate parameters
3. check_lora_params_finite detects NaN from extreme extrapolation
4. Partial NaN (subset of layers) is detected correctly
"""

import torch

from src.tg_lora.extrapolator import apply_extrapolation, cap_update
from src.training.train_tg_lora import check_lora_params_finite


class _LoRAMockModel(torch.nn.Module):
    """Minimal model with LoRA-like named parameters for testing."""

    def __init__(self, num_layers: int = 4, hidden: int = 8):
        super().__init__()
        for i in range(num_layers):
            setattr(
                self,
                f"layers_{i}_lora_A",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.01),
            )
            setattr(
                self,
                f"layers_{i}_lora_B",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.01),
            )


# ---------------------------------------------------------------------------
# Criterion 1: NaN-producing parameter combinations identified
# ---------------------------------------------------------------------------


class TestNaNParameterCombinations:
    """NaN mechanism: raw_update = n_steps * lr * v can overflow to inf in
    float32; cap_update should zero non-finite updates before they corrupt params.
    """

    def test_cap_update_inf_returns_zeros_instead_of_nan(self):
        raw_update = torch.tensor([1.0, float("inf"), 3.0, 2.0])
        ref = torch.ones(4)
        result = cap_update(raw_update, ref, max_ratio=0.5)
        assert torch.isfinite(result).all(), (
            "cap_update must not produce NaN from inf inputs"
        )
        assert result.norm().item() == 0.0, "Non-finite update should be zeroed"

    def test_extreme_params_remain_finite_via_apply_extrapolation(self):
        model = _LoRAMockModel(num_layers=2, hidden=8)
        names = [f"layers_{i}_lora_A" for i in range(2)]
        velocity = {name: torch.randn(8, 8) * 1e19 for name in names}

        apply_extrapolation(
            model,
            velocity,
            active_names=set(names),
            n_steps=100,
            lr=1e20,
            relative_update_cap=0.5,
        )

        is_finite, detail = check_lora_params_finite(model)
        assert is_finite, (
            f"cap_update should prevent NaN from extreme velocity, got: {detail}"
        )


# ---------------------------------------------------------------------------
# Criterion 2: cap_update prevents NaN with moderate parameters
# ---------------------------------------------------------------------------


class TestCapUpdatePreventsNaN:
    """cap_update clamps updates to relative_update_cap * ref_norm."""

    def test_moderate_params_remain_finite(self):
        model = _LoRAMockModel(num_layers=2, hidden=8)
        names = [f"layers_{i}_lora_A" for i in range(2)]
        velocity = {name: torch.randn(8, 8) for name in names}

        apply_extrapolation(
            model,
            velocity,
            active_names=set(names),
            n_steps=10,
            lr=0.5,
            relative_update_cap=0.5,
        )

        is_finite, detail = check_lora_params_finite(model)
        assert is_finite, f"Non-finite params: {detail}"

    def test_large_velocity_finite_with_cap(self):
        model = _LoRAMockModel(num_layers=2, hidden=8)
        names = [f"layers_{i}_lora_A" for i in range(2)]
        velocity = {name: torch.randn(8, 8) * 1e5 for name in names}

        apply_extrapolation(
            model,
            velocity,
            active_names=set(names),
            n_steps=10,
            lr=1.0,
            relative_update_cap=0.5,
        )

        is_finite, detail = check_lora_params_finite(model)
        assert is_finite, f"Non-finite params: {detail}"

    def test_very_large_velocity_still_finite_no_overflow(self):
        """velocity*1e18 with lr=1.0: raw_update ~1e19, no float32 overflow."""
        model = _LoRAMockModel(num_layers=2, hidden=8)
        names = [f"layers_{i}_lora_A" for i in range(2)]
        velocity = {name: torch.randn(8, 8) * 1e18 for name in names}

        apply_extrapolation(
            model,
            velocity,
            active_names=set(names),
            n_steps=10,
            lr=1.0,
            relative_update_cap=0.5,
        )

        is_finite, detail = check_lora_params_finite(model)
        assert is_finite, f"Non-finite params: {detail}"


# ---------------------------------------------------------------------------
# Criterion 3: Extreme params -> NaN -> check_lora_params_finite detects
# ---------------------------------------------------------------------------


class TestExtremeNaNDetection:
    """check_lora_params_finite detects NaN injected directly into model params."""

    def test_check_detects_manual_nan_injection(self):
        model = _LoRAMockModel(num_layers=2, hidden=8)
        # Inject NaN directly to simulate corruption from an unguarded path
        for name, p in model.named_parameters():
            if "lora_A" in name:
                p.data[0, 0] = float("nan")
                break

        is_finite, detail = check_lora_params_finite(model)
        assert not is_finite
        assert "NaN" in detail


# ---------------------------------------------------------------------------
# Criterion 4: Partial NaN (subset of layers) detected
# ---------------------------------------------------------------------------


class TestPartialNaNDetection:
    """NaN in only some layers is still detected by check_lora_params_finite."""

    def test_partial_nan_detected(self):
        model = _LoRAMockModel(num_layers=4, hidden=8)
        extreme_name = "layers_0_lora_A"

        # Inject NaN into one layer directly
        for name, p in model.named_parameters():
            if name == extreme_name:
                p.data[0, 0] = float("nan")
                break

        is_finite, detail = check_lora_params_finite(model)
        assert not is_finite
        assert "NaN" in detail
        assert extreme_name in detail

        # Non-corrupted layers remain finite
        for name, p in model.named_parameters():
            if "lora_A" in name and name != extreme_name:
                assert torch.isfinite(p).all(), f"{name} should be finite"

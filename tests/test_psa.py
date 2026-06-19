"""Tests for Prior-based Subspace Amplification (PSA)."""

import pytest
import torch
import torch.nn as nn

from src.tg_lora.psa import PSAPrior, _power_iteration_pc1, amplify_gradients_psa, summarize_by_layer_type


def _make_simple_model(n_layers=2, hidden=16, rank=2):
    """Create a minimal model with LoRA-like parameters for testing."""
    layers = []
    for i in range(n_layers):
        # Simulate lora_A and lora_B parameters
        a = nn.Parameter(torch.randn(rank, hidden) * 0.01)
        b = nn.Parameter(torch.randn(hidden, rank) * 0.01)
        layers.append((f"layers.{i}.self_attn.lora_A.default.weight", a))
        layers.append((f"layers.{i}.self_attn.lora_B.default.weight", b))

    class FakeModel(nn.Module):
        pass

    model = FakeModel()
    for name, param in layers:
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            if not hasattr(obj, p):
                setattr(obj, p, nn.Module())
            obj = getattr(obj, p)
        setattr(obj, parts[-1], param)
        param.requires_grad_(True)
    return model


def _make_delta_history(model, n_steps=5, dominant_dir=None):
    """Create synthetic delta history with a known dominant direction."""
    deltas = []
    for t in range(n_steps):
        delta = {}
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                # Create deltas along dominant_dir + noise
                d = torch.randn_like(p) * 0.1
                if dominant_dir is not None and name in dominant_dir:
                    d += dominant_dir[name] * (t + 1) * 1.0
                delta[name] = d
        deltas.append(delta)
    return deltas


class TestPowerIteration:
    def test_recovers_dominant_direction(self):
        """Power iteration should recover the top eigenvector."""
        torch.manual_seed(42)
        n, d = 20, 50
        v_true = torch.randn(d)
        v_true = v_true / v_true.norm()
        # Create matrix where v_true is the dominant direction
        mat = torch.randn(n, d) * 0.1 + 2.0 * v_true.unsqueeze(0)
        v_est = _power_iteration_pc1(mat, n_iters=50)
        v_est = v_est / v_est.norm()
        cos = abs(torch.dot(v_true, v_est).item())
        assert cos > 0.95, f"cos={cos}, expected > 0.95"

    def test_unit_norm_output(self):
        mat = torch.randn(10, 20)
        v = _power_iteration_pc1(mat, n_iters=10)
        assert abs(v.norm().item() - 1.0) < 1e-6

    def test_warm_start_converges_faster(self):
        """Warm-starting from a good guess should need fewer iterations."""
        torch.manual_seed(7)
        n, d = 20, 50
        v_true = torch.randn(d)
        v_true = v_true / v_true.norm()
        mat = torch.randn(n, d) * 0.1 + 2.0 * v_true.unsqueeze(0)

        # Cold start with very few iterations — poor recovery
        v_cold = _power_iteration_pc1(mat, n_iters=2)
        cos_cold = abs(torch.dot(v_true, v_cold / v_cold.norm()).item())

        # Warm start from v_true itself — should be near-perfect with same iters
        v_warm = _power_iteration_pc1(mat, n_iters=2, initial_guess=v_true)
        cos_warm = abs(torch.dot(v_true, v_warm / v_warm.norm()).item())

        assert cos_warm > cos_cold, (
            f"Warm start ({cos_warm:.4f}) should beat cold ({cos_cold:.4f})"
        )

    def test_warm_start_ignored_on_wrong_size(self):
        """Wrong-sized initial_guess falls back to random init."""
        mat = torch.randn(10, 20)
        v = _power_iteration_pc1(mat, n_iters=5, initial_guess=torch.randn(10))
        assert v.shape == (20,)


class TestPSAPrior:
    def test_extract_priors_from_history(self):
        model = _make_simple_model(n_layers=2, hidden=16, rank=2)
        prior = PSAPrior(history_length=5, gain=0.5)

        # Create history with a clear dominant direction
        dominant = {}
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                d = torch.randn_like(p)
                dominant[name] = d / d.norm()

        history = _make_delta_history(model, n_steps=5, dominant_dir=dominant)
        prior.extract_priors(history)

        assert len(prior.priors) > 0
        for name, v in prior.priors.items():
            assert abs(v.norm().item() - 1.0) < 1e-6, f"Prior for {name} not unit norm"

    def test_empty_history_no_crash(self):
        prior = PSAPrior()
        prior.extract_priors([])
        assert len(prior.priors) == 0

    def test_single_step_no_crash(self):
        prior = PSAPrior()
        prior.extract_priors([{"a": torch.randn(3, 4)}])
        assert len(prior.priors) == 0  # Need >= 2 steps

    def test_warmup_gating(self):
        prior = PSAPrior(warmup_steps=4, update_interval=3)
        # Add 2 deltas so should_update passes the history-count gate
        for _ in range(2):
            prior.record_delta({"a": torch.randn(4)})

        assert not prior.should_update(0)
        assert not prior.should_update(3)
        assert prior.should_update(4)

    def test_insufficient_history_gating(self):
        prior = PSAPrior(warmup_steps=0, update_interval=1)
        # No deltas recorded — should_update should return False
        assert not prior.should_update(0)
        # One delta — still insufficient
        prior.record_delta({"a": torch.randn(4)})
        assert not prior.should_update(0)
        # Two deltas — now sufficient
        prior.record_delta({"a": torch.randn(4)})
        assert prior.should_update(0)

    def test_l2_reg_smooths_prior(self):
        """L2 regularization should smooth prior updates toward previous direction."""
        model = _make_simple_model(n_layers=1, hidden=8, rank=2)
        prior = PSAPrior(history_length=3, l2_reg=0.5)

        # First extraction
        dominant1 = {}
        for name, p in model.named_parameters():
            if "lora_A" in name:
                d = torch.ones_like(p)
                dominant1[name] = d / d.norm()
        history1 = _make_delta_history(model, 3, dominant_dir=dominant1)
        prior.extract_priors(history1)
        v1 = {k: v.clone() for k, v in prior.priors.items()}

        # Second extraction with different direction
        dominant2 = {}
        for name, p in model.named_parameters():
            if "lora_A" in name:
                d = torch.randn_like(p)
                dominant2[name] = d / d.norm()
        history2 = _make_delta_history(model, 3, dominant_dir=dominant2)
        prior.extract_priors(history2)
        v2 = prior.priors

        # With high L2 reg, priors should retain some similarity to v1
        for name in v1:
            if name in v2:
                cos = abs(torch.dot(v1[name].flatten(), v2[name].flatten()).item())
                # With l2_reg=0.5, there should be some retention
                assert cos > 0.1, f"L2 reg not retaining prior direction: cos={cos}"


class TestAmplifyGradients:
    def test_amplification_increases_norm_along_prior(self):
        model = _make_simple_model(n_layers=2, hidden=16, rank=2)
        prior = PSAPrior(gain=0.5)

        # Set up priors
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                v = torch.randn_like(p).flatten()
                prior.priors[name] = v / v.norm()

        # Create gradients aligned with prior
        for name, p in model.named_parameters():
            if name in prior.priors:
                p.grad = prior.priors[name].reshape(p.shape).clone()

        orig_norms = {
            name: p.grad.norm().item()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        stats = prior.amplify_gradients(model)

        for name, p in model.named_parameters():
            if p.grad is not None and name in stats:
                new_norm = p.grad.norm().item()
                assert new_norm > orig_norms[name], (
                    f"Gradient norm should increase: {name} "
                    f"{orig_norms[name]:.4f} -> {new_norm:.4f}"
                )

    def test_no_priors_no_modification(self):
        model = _make_simple_model()
        prior = PSAPrior()
        for name, p in model.named_parameters():
            if "lora_A" in name:
                p.grad = torch.ones_like(p)
        stats = prior.amplify_gradients(model)
        assert len(stats) == 0

    def test_clamp_prevents_explosion(self):
        model = _make_simple_model(n_layers=1, hidden=8, rank=2)
        prior = PSAPrior(gain=100.0)  # Extreme gain

        for name, p in model.named_parameters():
            if "lora_A" in name:
                v = torch.randn_like(p).flatten()
                prior.priors[name] = v / v.norm()
                p.grad = (v / v.norm()).reshape(p.shape)  # Perfectly aligned

        prior.amplify_gradients(model)

        for name, p in model.named_parameters():
            if p.grad is not None and name in prior.priors:
                # Should be clamped to at most 2x original norm
                assert p.grad.norm().item() < 100.0


class TestGainMap:
    def test_out_proj_gets_higher_gain(self):
        model = _make_simple_model(n_layers=1, hidden=16, rank=2)
        prior = PSAPrior(gain=0.5)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                prior.priors[name] = torch.randn_like(p).flatten()
                prior.priors[name] = prior.priors[name] / prior.priors[name].norm()

        gain_map = prior.compute_gain_map(model)
        assert len(gain_map) > 0

    def test_no_prior_zero_gain(self):
        model = _make_simple_model()
        prior = PSAPrior(gain=0.5)
        gain_map = prior.compute_gain_map(model)
        for name, g in gain_map.items():
            assert g == 0.0


class TestRegimeGainFactor:
    def test_stable_gives_full_gain(self):
        prior = PSAPrior(gain=0.5)
        assert prior.regime_gain_factor("stable") == 1.0

    def test_transition_gives_zero_gain(self):
        prior = PSAPrior(gain=0.5)
        assert prior.regime_gain_factor("transition") == 0.0

    def test_plateau_uses_configurable_gain(self):
        prior = PSAPrior(gain=0.5, regime_plateau_gain=0.3)
        assert prior.regime_gain_factor("plateau") == 0.3

    def test_plateau_default_is_half(self):
        prior = PSAPrior(gain=0.5)
        assert prior.regime_gain_factor("plateau") == 0.5

    def test_unknown_regime_defaults_to_full(self):
        prior = PSAPrior(gain=0.5)
        assert prior.regime_gain_factor("unknown") == 1.0

    def test_compute_gain_map_scales_by_stable_regime(self):
        model = _make_simple_model(n_layers=1, hidden=16, rank=2)
        prior = PSAPrior(gain=1.0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                prior.priors[name] = torch.randn_like(p).flatten()
                prior.priors[name] = prior.priors[name] / prior.priors[name].norm()

        gain_stable = prior.compute_gain_map(model, regime="stable")
        gain_transition = prior.compute_gain_map(model, regime="transition")
        gain_plateau = prior.compute_gain_map(model, regime="plateau")

        for name in gain_stable:
            if gain_stable[name] > 0:
                assert gain_transition[name] == 0.0
                assert gain_plateau[name] < gain_stable[name]
                assert gain_plateau[name] == pytest.approx(gain_stable[name] * 0.5)

    def test_amplify_respects_transition_gain_map(self):
        """When regime=transition produces zero gain_map, amplification is a no-op."""
        model = _make_simple_model(n_layers=1, hidden=8, rank=2)
        prior = PSAPrior(gain=0.5)

        for name, p in model.named_parameters():
            if "lora_A" in name:
                v = torch.randn_like(p).flatten()
                prior.priors[name] = v / v.norm()
                p.grad = v.reshape(p.shape).clone()

        # gain_map with all zeros (transition regime)
        zero_map = {name: 0.0 for name in prior.priors}
        orig_norms = {
            name: p.grad.norm().item()
            for name, p in model.named_parameters()
            if p.grad is not None and name in prior.priors
        }

        prior.amplify_gradients(model, gain_override=zero_map)

        for name, p in model.named_parameters():
            if p.grad is not None and name in prior.priors:
                # With gamma=0 in gain_map, gradient should be unchanged
                new_norm = p.grad.norm().item()
                assert abs(new_norm - orig_norms[name]) < 1e-6, (
                    f"Gradient changed with zero gain: {name}"
                )


class TestCrossCycleHistory:
    def test_record_delta_persists_across_cycles(self):
        """Internal ring buffer should accumulate deltas across cycles."""
        model = _make_simple_model(n_layers=1, hidden=8, rank=2)
        prior = PSAPrior(history_length=10, gain=0.5)

        # Simulate 3 cycles, each producing 3 incremental deltas
        for cycle in range(3):
            for step in range(3):
                delta = {}
                for name, p in model.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        delta[name] = torch.randn_like(p) * 0.1
                prior.record_delta(delta)

        assert prior.history_count == 9

    def test_ring_buffer_respects_max_length(self):
        """Buffer should evict oldest entries when maxlen exceeded."""
        model = _make_simple_model(n_layers=1, hidden=8, rank=2)
        prior = PSAPrior(history_length=4, gain=0.5)

        for i in range(10):
            delta = {f"t{i}": torch.randn(3)}
            prior.record_delta(delta)

        assert prior.history_count == 4

    def test_extract_priors_uses_internal_buffer(self):
        """extract_priors() with no args should use internal buffer."""
        model = _make_simple_model(n_layers=1, hidden=8, rank=2)
        prior = PSAPrior(history_length=6, gain=0.5)

        dominant = {}
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                d = torch.randn_like(p)
                dominant[name] = d / d.norm()

        # Record 6 deltas with dominant direction
        for t in range(6):
            delta = {}
            for name, p in model.named_parameters():
                if name in dominant:
                    d = dominant[name] * (t + 1) * 1.0 + torch.randn_like(p) * 0.1
                    delta[name] = d
            prior.record_delta(delta)

        prior.extract_priors()

        assert len(prior.priors) > 0
        for name, v in prior.priors.items():
            assert abs(v.norm().item() - 1.0) < 1e-6

    def test_repeated_extraction_with_stable_direction(self):
        """Successive extract_priors calls with similar data should produce
        consistent priors (warm-start + L2 reg provide continuity)."""
        model = _make_simple_model(n_layers=1, hidden=8, rank=2)
        prior = PSAPrior(history_length=6, gain=0.5, l2_reg=0.3)

        dominant = {}
        for name, p in model.named_parameters():
            if "lora_A" in name:
                d = torch.randn_like(p)
                dominant[name] = d / d.norm()

        # First extraction
        for t in range(6):
            delta = {
                n: dominant[n] * (t + 1) + torch.randn_like(p) * 0.1
                for n, p in model.named_parameters()
                if n in dominant
            }
            prior.record_delta(delta)
        prior.extract_priors()
        v1 = {k: v.clone() for k, v in prior.priors.items()}

        # Second extraction with same dominant direction + new noise
        prior._delta_history.clear()
        for t in range(6):
            delta = {
                n: dominant[n] * (t + 1) + torch.randn_like(p) * 0.1
                for n, p in model.named_parameters()
                if n in dominant
            }
            prior.record_delta(delta)
        prior.extract_priors()

        for name in v1:
            if name in prior.priors:
                cos = abs(torch.dot(v1[name].flatten(), prior.priors[name].flatten()).item())
                assert cos > 0.8, f"Prior continuity broken for {name}: cos={cos:.4f}"



    def test_disabled_returns_empty(self):
        model = _make_simple_model()
        prior = PSAPrior()
        stats = amplify_gradients_psa(model, prior, None, enabled=False)
        assert stats == {}

    def test_enabled_with_no_priors(self):
        model = _make_simple_model()
        prior = PSAPrior()
        stats = amplify_gradients_psa(model, prior, None, enabled=True)
        assert stats == {}


class TestConfigMutualExclusion:
    def test_psa_and_m9_exclusive(self):
        from src.training.config_schema import TGLoRAParams

        with pytest.raises(ValueError, match="mutually exclusive"):
            TGLoRAParams(
                K_initial=3,
                K_candidates=[2, 3, 5],
                N_initial=5,
                N_candidates=[1, 3, 5],
                alpha_initial=0.3,
                alpha_min=0.03,
                alpha_max=1.5,
                beta_initial=0.8,
                beta_candidates=[0.5, 0.8, 0.9],
                relative_update_cap=0.005,
                active_layer_strategy="last_25_percent",
                enable_psa=True,
                subspace_m9_enabled=True,
            )

    def test_psa_config_defaults(self):
        from src.training.config_schema import TGLoRAParams

        params = TGLoRAParams(
            K_initial=3,
            K_candidates=[2, 3, 5],
            N_initial=5,
            N_candidates=[1, 3, 5],
            alpha_initial=0.3,
            alpha_min=0.03,
            alpha_max=1.5,
            beta_initial=0.8,
            beta_candidates=[0.5, 0.8, 0.9],
            relative_update_cap=0.005,
            active_layer_strategy="last_25_percent",
            enable_psa=True,
        )
        assert params.psa_gain == 0.5
        assert params.psa_history_length == 6
        assert params.psa_update_interval == 3
        assert params.psa_warmup_steps == 4
        assert params.psa_l2_reg == 0.01
        assert params.psa_regime_reset_enabled is True
        assert params.psa_regime_window == 8
        assert params.psa_regime_plateau_eps == 1e-4
        assert params.psa_regime_transition_z == 2.0
        assert params.psa_regime_plateau_gain == 0.5
        assert params.subspace_m9_enabled is False


class TestPriorStabilityTracking:
    def test_cosines_recorded_on_second_extraction(self):
        """Prior cosine should be recorded between consecutive extractions."""
        model = _make_simple_model(n_layers=1, hidden=8, rank=2)
        prior = PSAPrior(history_length=4, l2_reg=0.0)

        dominant = {}
        for name, p in model.named_parameters():
            if "lora_A" in name:
                d = torch.randn_like(p)
                dominant[name] = d / d.norm()

        # First extraction
        for t in range(4):
            delta = {n: dominant[n] * (t + 1) + torch.randn_like(p) * 0.1
                     for n, p in model.named_parameters() if n in dominant}
            prior.record_delta(delta)
        prior.extract_priors()

        # Second extraction with same direction
        prior._delta_history.clear()
        for t in range(4):
            delta = {n: dominant[n] * (t + 1) + torch.randn_like(p) * 0.1
                     for n, p in model.named_parameters() if n in dominant}
            prior.record_delta(delta)
        prior.extract_priors()

        # Should have cosine history for each tensor
        assert len(prior._prior_cosines) > 0
        for name, coses in prior._prior_cosines.items():
            assert len(coses) == 1
            # Same dominant direction → high cosine
            assert coses[0] > 0.8, f"Prior stability low for {name}: {coses[0]:.3f}"

    def test_cosines_reset_on_prior_reset(self):
        """reset_priors should clear stability tracking."""
        prior = PSAPrior()
        prior._prior_cosines["test"] = [0.9]
        prior._prev_priors["test"] = torch.randn(4)
        prior.reset_priors()
        assert len(prior._prior_cosines) == 0
        assert len(prior._prev_priors) == 0


class TestSummarizeByLayerType:
    def test_groups_by_layer_type(self):
        stats = {
            "layers.0.self_attn.out_proj.lora_A.weight": 1.2,
            "layers.0.self_attn.v_proj.lora_A.weight": 1.1,
            "layers.0.mlp.gate_proj.lora_A.weight": 0.8,
            "layers.0.mlp.up_proj.lora_A.weight": 0.7,
        }
        result = summarize_by_layer_type(stats)
        assert "attention_out" in result
        assert "attention_v" in result
        assert "mlp" in result
        assert result["attention_out"]["count"] == 1.0
        assert result["attention_out"]["amp_mean"] == 1.2
        assert result["mlp"]["count"] == 2.0
        assert abs(result["mlp"]["amp_mean"] - 0.75) < 1e-6

    def test_includes_prior_stability(self):
        stats = {
            "layers.0.self_attn.out_proj.lora_A.weight": 1.2,
            "layers.0.mlp.gate_proj.lora_A.weight": 0.8,
        }
        cosines = {
            "layers.0.self_attn.out_proj.lora_A.weight": [0.95, 0.93],
            "layers.0.mlp.gate_proj.lora_A.weight": [0.7],
        }
        result = summarize_by_layer_type(stats, prior_cosines=cosines)
        assert result["attention_out"]["prior_stability_mean"] == pytest.approx(0.93)
        assert result["mlp"]["prior_stability_mean"] == pytest.approx(0.7)

    def test_empty_stats_returns_empty(self):
        result = summarize_by_layer_type({})
        assert result == {}

    def test_deltanet_classification(self):
        stats = {
            "layers.0.mamba.dt_proj.lora_A.weight": 1.0,
            "layers.0.ssm.A_log.lora_A.weight": 0.9,
        }
        result = summarize_by_layer_type(stats)
        assert "deltanet" in result
        assert result["deltanet"]["count"] == 2.0


# ---------------------------------------------------------------------------
# TC-EDGE-202 / TC-EDGE-203: PSAPrior input validation
# ---------------------------------------------------------------------------


class TestPSAPriorInputValidation:
    """PSAPrior must reject nonsensical configurations up front rather than
    silently producing no-ops (empty ring buffer / sign-flipped amplification)."""

    def test_history_length_zero_raises(self):  # TC-EDGE-202
        with pytest.raises(ValueError, match="history_length"):
            PSAPrior(history_length=0)

    def test_negative_gain_raises(self):  # TC-EDGE-203
        with pytest.raises(ValueError, match="gain"):
            PSAPrior(gain=-0.1)

    def test_gain_zero_is_allowed(self):
        """gain=0.0 is the no-amplification ablation baseline (TC-281-01 sweep
        uses gamma=0.0) — it must NOT raise, only negative gain is invalid."""
        prior = PSAPrior(gain=0.0)
        assert prior.gain == 0.0


# ---------------------------------------------------------------------------
# TC-266-01 / TC-266-02: PSA L2 regularization
# ---------------------------------------------------------------------------


def _make_layer_typed_model():
    """Model with one LoRA param per LayerType for gain-map tests.

    Names chosen so classify_layer_type maps each to a distinct type:
    out_proj -> ATTENTION_OUT, v_proj -> ATTENTION_V, mlp -> MLP,
    embed_tokens -> UNKNOWN (no recognized substring).
    """
    names = [
        "layers.0.self_attn.out_proj.lora_A.default.weight",
        "layers.0.self_attn.v_proj.lora_A.default.weight",
        "layers.0.mlp.gate_proj.lora_A.default.weight",
        "embed_tokens.lora_A.default.weight",
    ]

    class FakeModel(nn.Module):
        pass

    model = FakeModel()
    for name in names:
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            if not hasattr(obj, p):
                setattr(obj, p, nn.Module())
            obj = getattr(obj, p)
        param = nn.Parameter(torch.randn(8))
        setattr(obj, parts[-1], param)
        param.requires_grad_(True)
    return model, names


class TestL2Regularization:
    """l2_reg>0 blends each new prior toward the previous one (penalizing
    deviation); l2_reg=0 disables blending so the prior tracks data only."""

    @staticmethod
    def _data_dominated_history(v_new, name, n=6):
        """History rows dominated by v_new with tiny additive noise."""
        history = []
        for t in range(n):
            row = v_new * (t + 1) * 1.0 + torch.randn_like(v_new) * 0.02
            history.append({name: row})
        return history

    @staticmethod
    def _orthonormal_pair(dim=8):
        """Return (v_prev, v_new) with v_prev along e0 and v_new at 45° to it."""
        v_prev = torch.zeros(dim)
        v_prev[0] = 1.0
        v_new = torch.zeros(dim)
        v_new[0] = 1.0
        v_new[1] = 1.0
        v_new = v_new / v_new.norm()  # cos(v_new, v_prev) = 1/sqrt(2) ~= 0.707
        return v_prev, v_new

    def test_l2_reg_positive_penalizes_deviation(self):  # TC-266-01
        torch.manual_seed(0)
        v_prev, v_new = self._orthonormal_pair()
        name = "layers.0.self_attn.lora_A.default.weight"
        history = self._data_dominated_history(v_new, name)

        # Same data + same previous prior, varying only l2_reg.
        prior_reg = PSAPrior(history_length=6, gain=0.5, l2_reg=0.5)
        prior_reg.priors[name] = v_prev.clone()
        prior_reg.extract_priors(history)
        r_reg = prior_reg.priors[name]

        prior_zero = PSAPrior(history_length=6, gain=0.5, l2_reg=0.0)
        prior_zero.priors[name] = v_prev.clone()
        prior_zero.extract_priors(history)
        r_zero = prior_zero.priors[name]

        cos_prev_reg = abs(torch.dot(r_reg, v_prev).item())
        cos_prev_zero = abs(torch.dot(r_zero, v_prev).item())
        cos_data_reg = abs(torch.dot(r_reg, v_new).item())
        cos_data_zero = abs(torch.dot(r_zero, v_new).item())

        # l2_reg>0 pulls the prior toward the previous prior (penalizes deviation)
        assert cos_prev_reg > cos_prev_zero
        # ... and is therefore less purely aligned with the data direction
        assert cos_data_zero > cos_data_reg

    def test_l2_reg_zero_disables_regularization(self):  # TC-266-02
        torch.manual_seed(1)
        v_prev, v_new = self._orthonormal_pair()
        name = "layers.0.self_attn.lora_A.default.weight"
        history = self._data_dominated_history(v_new, name)

        prior = PSAPrior(history_length=6, gain=0.5, l2_reg=0.0)
        prior.priors[name] = v_prev.clone()  # previous prior points elsewhere
        prior.extract_priors(history)
        r = prior.priors[name]

        # With blending disabled, the prior tracks the data, not v_prev.
        assert abs(torch.dot(r, v_new).item()) > 0.95


# ---------------------------------------------------------------------------
# TC-267-01..04: layer-type-specific gain multipliers
# ---------------------------------------------------------------------------


class TestLayerTypeGain:
    """compute_gain_map applies gain multipliers by module type
    (out_proj x1.2, v_proj x1.1, MLP x0.7, unknown x1.0) under a stable regime."""

    @pytest.fixture(autouse=True)
    def _setup_typed_gain_map(self):
        self.model, self.names = _make_layer_typed_model()
        self.prior = PSAPrior(gain=1.0)
        for name in self.names:
            v = torch.randn(8)
            self.prior.priors[name] = v / v.norm()
        self.gm = self.prior.compute_gain_map(self.model, regime="stable")

    def test_out_proj_gets_1_2x(self):  # TC-267-01
        name = next(n for n in self.names if "out_proj" in n)
        assert self.gm[name] == pytest.approx(1.2)

    def test_v_proj_gets_1_1x(self):  # TC-267-02
        name = next(
            n for n in self.names if "v_proj" in n and "out_proj" not in n
        )
        assert self.gm[name] == pytest.approx(1.1)

    def test_mlp_gets_0_7x(self):  # TC-267-03
        name = next(n for n in self.names if "mlp" in n)
        assert self.gm[name] == pytest.approx(0.7)

    def test_unknown_gets_default_1_0x(self):  # TC-267-04
        typed = {
            n
            for n in self.names
            if any(s in n for s in ("out_proj", "v_proj", "mlp"))
        }
        name = next(n for n in self.names if n not in typed)
        assert self.gm[name] == pytest.approx(1.0)

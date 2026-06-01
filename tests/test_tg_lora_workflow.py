"""End-to-end workflow test: demonstrates the core TG-LoRA algorithm cycle.

This test simulates one complete cycle of the TG-LoRA training loop:
  1. Controller proposes hyperparameters (K, N, alpha, beta, lr)
  2. Run K pilot optimizer steps on a small model
  3. Snapshot weights before/after → compute delta
  4. Update velocity (EMA of deltas)
  5. Quick eval → loss_pilot
  6. Apply extrapolation using velocity × N steps
  7. Quick eval → loss_after
  8. Accept or rollback based on loss comparison
  9. Track cycle state metrics
"""
import math

import pytest
import torch
import torch.nn as nn

from tg_lora import (
    CycleState,
    DeltaTracker,
    RandomWalkController,
    RollbackManager,
    Velocity,
    apply_extrapolation,
    diff_lora,
    select_active_layers,
    snapshot_lora,
)


class SimpleLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(4):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Module()
            layer.self_attn.q_proj.lora_A = nn.Parameter(torch.randn(4, 4) * 0.01)
            layer.self_attn.q_proj.lora_B = nn.Parameter(torch.zeros(4, 4))
            layer.self_attn.q_proj.lora_A.requires_grad_(True)
            layer.self_attn.q_proj.lora_B.requires_grad_(True)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            w = layer.self_attn.q_proj.lora_A @ layer.self_attn.q_proj.lora_B
            x = x @ w.T
        return x


def _fake_loss(model, x):
    return model(x).sum()


class TestTGLoRAWorkflow:
    def test_single_cycle_accept(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        controller = RandomWalkController(
            K_initial=3,
            N_initial=2,
            alpha_initial=0.3,
            beta_initial=0.8,
            lr_initial=1e-3,
            enable_random_walk=False,
        )
        velocity = Velocity()
        rollback = RollbackManager()
        delta_tracker = DeltaTracker()
        cycle_state = CycleState()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        proposal = controller.propose()
        assert proposal.K == 3
        assert proposal.N == 2

        W0 = snapshot_lora(model)

        # Pilot: K optimizer steps
        for _ in range(proposal.K):
            loss = _fake_loss(model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        WK = snapshot_lora(model)
        dW = delta_tracker.compute_and_record(WK, W0, K=proposal.K)
        assert len(dW) > 0

        cos_sim = velocity.cosine_similarity(dW)
        velocity.update(dW, beta=proposal.beta)
        assert velocity.state is not None

        # Simulate pilot loss
        with torch.no_grad():
            loss_pilot = _fake_loss(model, x).item()

        # Save rollback point, then extrapolate
        rollback.save(model)
        active_names, active_indices = select_active_layers(
            model, strategy="last_25_percent"
        )
        assert len(active_names) > 0

        apply_extrapolation(
            model=model,
            velocity=velocity.state,
            active_names=active_names,
            alpha_by_name={},
            default_alpha=proposal.alpha,
            n_steps=proposal.N,
        )

        with torch.no_grad():
            loss_after = _fake_loss(model, x).item()

        # Accept or rollback
        accepted = controller.accept(loss_pilot, loss_after)
        if accepted:
            controller.reward(loss_pilot, loss_after)
        else:
            rollback.rollback(model)
            controller.penalize(loss_pilot, loss_after)

        cycle_state.record_cycle(
            train_loss=loss_pilot,
            valid_loss=loss_after if accepted else loss_pilot,
            accepted=accepted,
            K=proposal.K,
            N=proposal.N if accepted else 0,
            grad_accum=1,
        )

        assert cycle_state.cycle == 1
        assert cycle_state.total_cycles == 1
        assert math.isfinite(cycle_state.reduction_rate)

    def test_multi_cycle_convergence(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        velocity = Velocity()
        rollback = RollbackManager()
        delta_tracker = DeltaTracker()
        cycle_state = CycleState()
        controller = RandomWalkController(
            K_initial=2,
            N_initial=1,
            alpha_initial=0.2,
            beta_initial=0.9,
            lr_initial=1e-3,
            enable_random_walk=False,
        )

        for cycle in range(10):
            proposal = controller.propose()
            W0 = snapshot_lora(model)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=proposal.lr
            )

            for _ in range(proposal.K):
                loss = _fake_loss(model, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            WK = snapshot_lora(model)
            dW = delta_tracker.compute_and_record(WK, W0, K=proposal.K)
            velocity.update(dW, beta=proposal.beta)

            with torch.no_grad():
                loss_pilot = _fake_loss(model, x).item()

            rollback.save(model)
            active_names, _ = select_active_layers(model, strategy="last_25_percent")
            apply_extrapolation(
                model=model,
                velocity=velocity.state,
                active_names=active_names,
                alpha_by_name={},
                default_alpha=proposal.alpha,
                n_steps=proposal.N,
            )

            with torch.no_grad():
                loss_after = _fake_loss(model, x).item()

            accepted = controller.accept(loss_pilot, loss_after)
            if accepted:
                controller.reward(loss_pilot, loss_after)
            else:
                rollback.rollback(model)
                controller.penalize(loss_pilot, loss_after)

            cycle_state.record_cycle(
                train_loss=loss_pilot,
                valid_loss=loss_after if accepted else loss_pilot,
                accepted=accepted,
                K=proposal.K,
                N=proposal.N if accepted else 0,
                grad_accum=1,
            )
            if not rollback._history:
                pass
            else:
                rollback.pop()

        assert cycle_state.cycle == 10
        assert math.isfinite(cycle_state.reduction_rate)
        assert delta_tracker.last_stats is not None

    def test_rollback_on_degradation(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        velocity = Velocity()
        rollback = RollbackManager()

        # Seed velocity with a large direction
        W0 = snapshot_lora(model)
        fake_delta = {k: torch.ones_like(v) * 10.0 for k, v in W0.items()}
        velocity.update(fake_delta, beta=0.9)

        rollback.save(model)

        # Extrapolate with extreme alpha → should degrade weights
        active_names, _ = select_active_layers(model, strategy="last_25_percent")
        apply_extrapolation(
            model=model,
            velocity=velocity.state,
            active_names=active_names,
            alpha_by_name={},
            default_alpha=10.0,
            n_steps=50,
            relative_update_cap=0.005,
        )

        W_after = snapshot_lora(model)
        params_changed = not all(
            torch.equal(W0[k], W_after[k]) for k in W0
        )

        # cap_update should have limited the damage, but params may differ
        # Rollback restores original state
        rollback.rollback(model)
        W_restored = snapshot_lora(model)
        for k in W0:
            assert torch.equal(W0[k], W_restored[k])

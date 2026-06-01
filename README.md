# TG-LoRA

Velocity-based extrapolation for efficient LoRA fine-tuning.

TG-LoRA reduces backward-pass cost by predicting future LoRA weight updates from a running velocity (exponential moving average of past deltas). After a short pilot phase of K real gradient steps, it extrapolates N additional "free" steps — then accepts or rolls back based on validation loss.

## Install

```bash
pip install -e .
```

Requires Python 3.11+, PyTorch, and a CUDA GPU.

## Algorithm

Each **cycle** repeats:

```
1. Propose (K, N, alpha, beta, lr)     ← controller
2. Snapshot W0
3. Run K real optimizer steps → WK
4. Compute delta dW = WK − W0
5. Update velocity v ← EMA(v, dW, beta)
6. Eval → loss_pilot
7. Extrapolate: W ← WK + N × alpha × v   (active layers only)
8. Eval → loss_after
9. If loss_after ≤ loss_pilot + tol → accept, reward
   Else → rollback to WK, penalize
```

The **reduction rate** measures how many backward passes were replaced by cheap extrapolation:

```
reduction_rate = 1 − (real_backward_passes / total_equivalent_passes)
```

## Quick Start

```python
import torch
from tg_lora import (
    RandomWalkController,
    Velocity,
    RollbackManager,
    DeltaTracker,
    CycleState,
    apply_extrapolation,
    select_active_layers,
    snapshot_lora,
)

# Set up controller, velocity tracker, rollback manager
controller = RandomWalkController(K_initial=3, N_initial=5, alpha_initial=0.3, beta_initial=0.8)
velocity = Velocity()
rollback = RollbackManager()
delta_tracker = DeltaTracker()
cycle_state = CycleState()

for cycle in range(max_cycles):
    proposal = controller.propose()   # (K, N, alpha, beta, lr)

    # 1. Snapshot before pilot
    W0 = snapshot_lora(model)

    # 2. K real optimizer steps
    optimizer = torch.optim.AdamW(trainable_params, lr=proposal.lr)
    for _ in range(proposal.K):
        loss = forward_backward(model, batch)
        optimizer.step()

    # 3. Compute delta and update velocity
    WK = snapshot_lora(model)
    dW = delta_tracker.compute_and_record(WK, W0, K=proposal.K)
    velocity.update(dW, beta=proposal.beta)

    # 4. Eval pilot
    loss_pilot = eval_loss(model, valid_loader)

    # 5. Save rollback point, then extrapolate
    rollback.save(model)
    active_names, _ = select_active_layers(model, strategy="last_25_percent")
    apply_extrapolation(model, velocity.state, active_names,
                        default_alpha=proposal.alpha, n_steps=proposal.N)

    # 6. Eval after extrapolation
    loss_after = eval_loss(model, valid_loader)

    # 7. Accept or rollback
    if loss_after <= loss_pilot + tolerance:
        controller.reward(loss_pilot, loss_after)
    else:
        rollback.rollback(model)
        controller.penalize(loss_pilot, loss_after)
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 3 | Real optimizer steps per cycle (pilot phase) |
| `N` | 5 | Extrapolation steps after pilot |
| `alpha` | 0.3 | Extrapolation step size |
| `beta` | 0.8 | EMA momentum for velocity |
| `relative_update_cap` | 0.005 | Max extrapolation magnitude relative to weight norm |
| `rollback_tolerance` | 0.005 | Accept loss increase up to this relative threshold |

The `RandomWalkController` adapts K, N, alpha, beta, and lr across cycles based on accept/reject history. Set `enable_random_walk=False` for fixed hyperparameters.

## Layer Sampling Strategies

| Strategy | Description |
|----------|-------------|
| `last_25_percent` | Extrapolate only the last 25% of decoder layers |
| `last_25_percent_plus_random_2` | Last 25% + 2 random middle layers (default) |
| `middle_random` | Random 1/3 of all layers |
| `lisa_like_weighted` | Score-based weighted sampling |

## Package API

```python
from tg_lora import (
    # Core algorithm
    Velocity,                   # EMA velocity tracker
    apply_extrapolation,        # Apply velocity-based weight extrapolation
    cap_update,                 # Cap update magnitude relative to reference

    # Training loop support
    CycleState,                 # Track cycle metrics (reduction rate, acceptance rate)
    DeltaTracker,               # Track weight change statistics
    RollbackManager,            # Save/restore LoRA weight snapshots
    RandomWalkController,       # Adaptive hyperparameter exploration

    # Layer selection
    select_active_layers,       # Choose which layers to extrapolate

    # Snapshot utilities
    snapshot_lora,              # Capture LoRA parameter state dict
    load_lora_snapshot,         # Restore LoRA parameters from snapshot
    diff_lora,                  # Compute difference between two snapshots

    # Trajectory analysis
    TrajectoryAnalyzer,         # Predict convergence, early-stop advice
    TrajectoryPoint,            # Single trajectory data point
)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT

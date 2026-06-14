"""Chunked checkpoint patch for MLX-LM Qwen3.5 GatedDelta training.

Qwen3.5 linear-attention layers use a custom recurrent Metal kernel for
inference, but the training path falls back to an ops-based Python loop so MLX
can differentiate it. For long sequences that loop retains every per-token
recurrent state in the backward graph. Chunking the recurrence and checkpointing
each chunk keeps only chunk-boundary states live.
"""

from __future__ import annotations

import os
from typing import Optional

import mlx.core as mx


_DEFAULT_CHUNK_SIZE = 512


def _chunk_size() -> int:
    raw = os.environ.get("MLX_GATED_DELTA_CHUNK", str(_DEFAULT_CHUNK_SIZE))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("MLX_GATED_DELTA_CHUNK must be an integer") from exc


def install() -> None:
    """Install the local MLX-LM gated-delta training patch."""
    import mlx_lm.models.gated_delta as gated_delta

    if getattr(gated_delta, "_tg_lora_chunked_checkpoint_installed", False):
        return

    original_update = gated_delta.gated_delta_update
    original_ops = gated_delta.gated_delta_ops
    original_kernel = gated_delta.gated_delta_kernel
    compute_g = gated_delta.compute_g

    def chunk_no_mask_ops(q, k, v, g, beta, state):
        return original_ops(q, k, v, g, beta, state, None)

    def chunk_with_mask_ops(q, k, v, g, beta, state, mask):
        return original_ops(q, k, v, g, beta, state, mask)

    @mx.custom_function
    def chunk_no_mask(q, k, v, g, beta, state):
        if mx.default_device() == mx.gpu and mx.metal.is_available():
            return original_kernel(q, k, v, g, beta, state, None)
        return original_ops(q, k, v, g, beta, state, None)

    @chunk_no_mask.vjp
    def chunk_no_mask_vjp(primals, cotangents, _outputs):
        _, vjps = mx.vjp(chunk_no_mask_ops, primals, cotangents)
        return vjps

    @mx.custom_function
    def chunk_with_mask(q, k, v, g, beta, state, mask):
        if mx.default_device() == mx.gpu and mx.metal.is_available():
            return original_kernel(q, k, v, g, beta, state, mask)
        return original_ops(q, k, v, g, beta, state, mask)

    @chunk_with_mask.vjp
    def chunk_with_mask_vjp(primals, cotangents, _outputs):
        q, k, v, g, beta, state, mask = primals

        def masked_ops(q, k, v, g, beta, state):
            return original_ops(q, k, v, g, beta, state, mask)

        _, vjps = mx.vjp(masked_ops, (q, k, v, g, beta, state), cotangents)
        return (*vjps, mx.zeros_like(mask))

    def chunked_ops(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        chunk = _chunk_size()
        if chunk <= 0:
            return original_ops(q, k, v, g, beta, state, mask)

        bsz, seq_len, num_k_heads, key_dim = q.shape
        num_v_heads, value_dim = v.shape[-2:]
        if state is None:
            state = mx.zeros(
                (bsz, num_v_heads, value_dim, key_dim),
                dtype=mx.float32,
            )

        ys = []
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            q_chunk = q[:, start:end]
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            g_chunk = g[:, start:end]
            beta_chunk = beta[:, start:end]
            if mask is None:
                y_chunk, state = chunk_no_mask(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    g_chunk,
                    beta_chunk,
                    state,
                )
            else:
                y_chunk, state = chunk_with_mask(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    g_chunk,
                    beta_chunk,
                    state,
                    mask[:, start:end],
                )
            ys.append(y_chunk)

        if len(ys) == 1:
            return ys[0], state
        return mx.concatenate(ys, axis=1), state

    def chunked_update(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        a: mx.array,
        b: mx.array,
        a_log: mx.array,
        dt_bias: mx.array,
        state: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        use_kernel: bool = True,
    ) -> tuple[mx.array, mx.array]:
        beta = mx.sigmoid(b)
        g = compute_g(a_log, a, dt_bias)
        if state is None:
            bsz, _, _, key_dim = q.shape
            num_v_heads, value_dim = v.shape[-2:]
            state = mx.zeros(
                (bsz, num_v_heads, value_dim, key_dim),
                dtype=mx.float32,
            )

        if use_kernel and mx.default_device() == mx.gpu and mx.metal.is_available():
            return original_kernel(q, k, v, g, beta, state, mask)
        return chunked_ops(q, k, v, g, beta, state, mask)

    gated_delta.gated_delta_update = chunked_update
    gated_delta._tg_lora_original_gated_delta_update = original_update
    gated_delta._tg_lora_chunked_checkpoint_installed = True

    try:
        import mlx_lm.models.qwen3_5 as qwen3_5
    except Exception:
        return
    qwen3_5.gated_delta_update = chunked_update

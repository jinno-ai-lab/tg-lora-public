import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.src.utils.gated_delta_patch import install  # noqa: E402


def test_chunked_gated_delta_matches_original_forward_and_grad(monkeypatch):
    monkeypatch.setenv("MLX_GATED_DELTA_CHUNK", "2")

    import mlx_lm.models.gated_delta as gated_delta

    original_update = gated_delta.gated_delta_update
    install()
    patched_update = gated_delta.gated_delta_update

    q = mx.random.normal((1, 5, 1, 32), dtype=mx.float32)
    k = mx.random.normal((1, 5, 1, 32), dtype=mx.float32)
    v = mx.random.normal((1, 5, 1, 3), dtype=mx.float32)
    a = mx.random.normal((1, 5, 1), dtype=mx.float32)
    b = mx.random.normal((1, 5, 1), dtype=mx.float32)
    a_log = mx.random.normal((1,), dtype=mx.float32)
    dt_bias = mx.zeros((1,), dtype=mx.float32)

    y_orig, state_orig = original_update(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        use_kernel=False,
    )
    y_patch, state_patch = patched_update(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        use_kernel=False,
    )
    mx.eval(y_orig, state_orig, y_patch, state_patch)
    assert bool(mx.allclose(y_orig, y_patch, atol=1e-5, rtol=1e-5).item())
    assert bool(mx.allclose(state_orig, state_patch, atol=1e-5, rtol=1e-5).item())

    def original_loss(q_in):
        y, state = original_update(
            q_in,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            use_kernel=False,
        )
        return mx.sum(y) + mx.sum(state)

    def patched_loss(q_in):
        y, state = patched_update(
            q_in,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            use_kernel=False,
        )
        return mx.sum(y) + mx.sum(state)

    grad_orig = mx.grad(original_loss)(q)
    grad_patch = mx.grad(patched_loss)(q)
    mx.eval(grad_orig, grad_patch)
    assert bool(mx.allclose(grad_orig, grad_patch, atol=1e-5, rtol=1e-5).item())

from __future__ import annotations

import pytest

from mlx.src.utils.shape_guard import SHAPE_ELEM_MAX, check_shape_dim, install


mx = pytest.importorskip("mlx.core")


def test_check_shape_dim_rejects_int32_overflow() -> None:
    with pytest.raises(OverflowError, match="32-bit integers"):
        check_shape_dim(SHAPE_ELEM_MAX + 1, "flatten")


def test_mlx_flatten_shape_product_overflow_is_guarded() -> None:
    install()
    array = mx.zeros((1 << 30, 2))

    with pytest.raises(OverflowError, match=r"\[flatten\]"):
        mx.flatten(array)


def test_mlx_reshape_inferred_shape_overflow_is_guarded() -> None:
    install()
    array = mx.zeros((1 << 30, 2))

    with pytest.raises(OverflowError, match=r"\[reshape\]"):
        mx.reshape(array, (-1,))


def test_mlx_reshape_explicit_shape_product_overflow_is_guarded() -> None:
    install()
    array = mx.zeros((1 << 30, 2))

    with pytest.raises(OverflowError, match=r"\[reshape\]"):
        mx.reshape(array, (1 << 30, 2))


def test_mlx_take_internal_flatten_overflow_is_guarded() -> None:
    install()
    array = mx.zeros((1 << 30, 2))
    indices = mx.array([0], mx.uint32)

    with pytest.raises(OverflowError, match=r"\[take\]"):
        mx.take(array, indices)


def test_mlx_shape_guard_keeps_safe_flatten_working() -> None:
    install()
    array = mx.zeros((2, 3))

    flattened = mx.flatten(array)

    assert flattened.shape == (6,)


def test_mlx_array_bound_methods_are_guarded() -> None:
    install()
    array = mx.zeros((1 << 30, 2))

    with pytest.raises(OverflowError, match=r"\[flatten\]"):
        array.flatten()

    with pytest.raises(OverflowError, match=r"\[reshape\]"):
        array.reshape(-1)

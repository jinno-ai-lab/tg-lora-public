"""Runtime guards for MLX int32 shape-product overflow.

MLX releases before PR #3524 can silently wrap shape dimensions produced by
operations such as flatten/reshape/take when the computed dimension exceeds the
int32 ShapeElem range. The wrapped shape can then drive impossible allocations.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Any


SHAPE_ELEM_MIN = -(2**31)
SHAPE_ELEM_MAX = 2**31 - 1


def _overflow_message(dim: int, op: str) -> str:
    prefix = f"[{op}] " if op else ""
    return (
        f"{prefix}Shape dimension {dim} is outside the supported range "
        f"[{SHAPE_ELEM_MIN}, {SHAPE_ELEM_MAX}]. MLX currently uses 32-bit "
        "integers for shape dimensions."
    )


def check_shape_dim(dim: int, op: str = "") -> int:
    """Return ``dim`` if it fits MLX ShapeElem, otherwise raise OverflowError."""
    value = int(dim)
    if value < SHAPE_ELEM_MIN or value > SHAPE_ELEM_MAX:
        raise OverflowError(_overflow_message(value, op))
    return value


def _normalize_axis(axis: int, ndim: int) -> int:
    value = int(axis)
    if value < 0:
        value += ndim
    return value


def _shape_product(shape: Iterable[int], op: str) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
        check_shape_dim(product, op)
    return product


def _array_shape(array: Any) -> tuple[int, ...] | None:
    shape = getattr(array, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _array_size(array: Any) -> int | None:
    size = getattr(array, "size", None)
    if size is not None:
        return int(size)
    shape = _array_shape(array)
    if shape is None:
        return None
    product = 1
    for dim in shape:
        product *= dim
    return product


def _get_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    idx: int,
    name: str,
    default: Any = None,
) -> Any:
    if len(args) > idx:
        return args[idx]
    return kwargs.get(name, default)


def _guard_flatten(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    array = _get_arg(args, kwargs, 0, "a")
    shape = _array_shape(array)
    if shape is None:
        return
    ndim = len(shape)
    if ndim == 0:
        return
    start_axis = _normalize_axis(_get_arg(args, kwargs, 1, "start_axis", 0), ndim)
    end_axis = _normalize_axis(_get_arg(args, kwargs, 2, "end_axis", -1), ndim)
    if start_axis < 0 or end_axis >= ndim or start_axis > end_axis:
        return
    _shape_product(shape[start_axis : end_axis + 1], "flatten")


def _guard_reshape(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    array = _get_arg(args, kwargs, 0, "a")
    requested = _get_arg(args, kwargs, 1, "shape")
    if requested is None:
        return
    if isinstance(requested, int):
        target_shape = [requested]
    else:
        target_shape = [int(dim) for dim in requested]

    infer_idx = None
    known_product = 1
    for idx, dim in enumerate(target_shape):
        if dim == -1:
            if infer_idx is not None:
                return
            infer_idx = idx
            continue
        check_shape_dim(dim, "reshape")
        known_product *= dim
        check_shape_dim(known_product, "reshape")

    if infer_idx is None:
        return

    input_size = _array_size(array)
    if input_size is None or known_product == 0:
        return
    inferred = input_size // known_product
    check_shape_dim(inferred, "reshape")


def _guard_take(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    array = _get_arg(args, kwargs, 0, "a")
    axis = _get_arg(args, kwargs, 2, "axis", None)
    if axis is not None:
        return
    shape = _array_shape(array)
    if shape is None:
        return
    _shape_product(shape, "take")


def _guard_array_flatten(
    array: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    _guard_flatten((array, *args), kwargs)


def _guard_array_reshape(
    array: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    if len(args) == 1 and not isinstance(args[0], int):
        requested = args[0]
    else:
        requested = args
    _guard_reshape((array, requested), kwargs)


def _guard_output_size(output: Any, op: str) -> None:
    size = _array_size(output)
    if size is not None:
        check_shape_dim(size, op)


def install() -> None:
    """Install PR #3524-style guards around Python-visible MLX shape ops."""
    import mlx.core as mx

    if getattr(mx, "_tg_lora_shape_guard_installed", False):
        return

    originals: dict[str, Any] = {}

    def wrap_pre(name: str, guard: Any) -> None:
        original = getattr(mx, name)
        originals[name] = original

        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            guard(args, kwargs)
            return original(*args, **kwargs)

        setattr(mx, name, guarded)

    def wrap_post(name: str, op: str) -> None:
        original = getattr(mx, name)
        originals[name] = original

        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            _guard_output_size(output, op)
            return output

        setattr(mx, name, guarded)

    wrap_pre("flatten", _guard_flatten)
    wrap_pre("reshape", _guard_reshape)
    wrap_pre("take", _guard_take)

    original_array_reshape = mx.array.reshape
    originals["array.reshape"] = original_array_reshape

    @wraps(original_array_reshape)
    def guarded_array_reshape(array: Any, *args: Any, **kwargs: Any) -> Any:
        _guard_array_reshape(array, args, kwargs)
        return original_array_reshape(array, *args, **kwargs)

    mx.array.reshape = guarded_array_reshape

    original_array_flatten = mx.array.flatten
    originals["array.flatten"] = original_array_flatten

    @wraps(original_array_flatten)
    def guarded_array_flatten(array: Any, *args: Any, **kwargs: Any) -> Any:
        _guard_array_flatten(array, args, kwargs)
        return original_array_flatten(array, *args, **kwargs)

    mx.array.flatten = guarded_array_flatten

    for name in ("conv_general", "conv1d", "conv2d", "conv3d"):
        if hasattr(mx, name):
            wrap_post(name, "conv")

    setattr(mx, "_tg_lora_shape_guard_originals", originals)
    setattr(mx, "_tg_lora_shape_guard_installed", True)

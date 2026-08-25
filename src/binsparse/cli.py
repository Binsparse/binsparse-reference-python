"""Command-line adapters for the Binsparse compliance tests."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_binsparse, save_binsparse
from .reformat import binsparse_to_coo, coo_to_binsparse, reformat
from .tensor import (
    _PREDEFINED_LEVELS,
    BinsparseLevel,
    BinsparseTensor,
    CustomTensor,
    DenseLevel,
    ElementLevel,
    SparseLevel,
)


def npy_to_binsparse_main(argv: Sequence[str] | None = None) -> int:
    """Implement ``npy_to_binsparse``."""
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 5:
        print(
            "usage: npy_to_binsparse <tensor_in> <pattern_in> <fill_value_in> "
            "<header_in> <tensor_out>",
            file=sys.stderr,
        )
        return 2
    tensor_in, pattern_in, fill_value_in, header_in, tensor_out = args

    dense = np.load(tensor_in, allow_pickle=False)
    pattern = np.asarray(np.load(pattern_in, allow_pickle=False), dtype=bool)
    fill_value = _load_fill_value(fill_value_in)
    header = _load_header(header_in)

    tensor = _tensor_from_npy(dense, pattern, fill_value, header)
    save_binsparse(tensor, tensor_out, header=header)
    return 0


def binsparse_to_npy_main(argv: Sequence[str] | None = None) -> int:
    """Implement ``binsparse_to_npy``."""
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 4:
        print(
            "usage: binsparse_to_npy <tensor_in> <tensor_out> "
            "<pattern_out> <fill_value_out>",
            file=sys.stderr,
        )
        return 2
    tensor_in, tensor_out, pattern_out, fill_value_out = args

    tensor = load_binsparse(tensor_in, alias=False)
    assert isinstance(tensor, CustomTensor)
    dense, pattern, fill_value = _npy_from_tensor(tensor)
    np.save(tensor_out, dense)
    np.save(pattern_out, pattern)
    np.save(fill_value_out, np.asarray(fill_value))
    return 0


def binsparse_to_binsparse_main(argv: Sequence[str] | None = None) -> int:
    """Implement ``binsparse_to_binsparse``."""
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 2:
        print(
            "usage: binsparse_to_binsparse <tensor_in> <tensor_out>",
            file=sys.stderr,
        )
        return 2
    tensor_in, tensor_out = args

    tensor = load_binsparse(tensor_in)
    save_binsparse(tensor, tensor_out)
    return 0


def _load_header(path: str) -> dict[str, Any]:
    with Path(path).open() as file:
        header = json.load(file)
    if not isinstance(header, dict):
        raise TypeError("header_in must contain a JSON object")
    return header


def _load_fill_value(path: str) -> Any:
    fill_value = np.load(path, allow_pickle=False)
    if fill_value.size != 1:
        raise ValueError("fill_value_in must contain exactly one value")
    return fill_value.reshape(-1)[0]


def _tensor_from_npy(
    dense: np.ndarray,
    pattern: np.ndarray,
    fill_value: Any,
    header: dict[str, Any],
) -> BinsparseTensor:
    if dense.shape != pattern.shape:
        raise ValueError("tensor_in and pattern_in must have the same shape")
    format, transpose = _format_from_header(header)
    target_transpose = transpose or tuple(range(dense.ndim))
    stored_shape = tuple(dense.shape[dimension] for dimension in target_transpose)

    entries = [
        (
            tuple(int(coord[dimension]) for dimension in target_transpose),
            dense[tuple(coord)],
        )
        for coord in np.argwhere(pattern)
    ]
    custom = coo_to_binsparse(format, entries, shape=stored_shape)
    custom.shape = tuple(dense.shape)
    custom.transpose = None if transpose is None else tuple(transpose)
    custom.fill = True
    custom.fill_value = fill_value
    return reformat(custom, header)


def _format_from_header(
    header: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, ...] | None]:
    format_name = header["format"]

    if format_name == "custom":
        custom = header.get("custom")
        if not isinstance(custom, dict) or "level" not in custom:
            raise ValueError("custom format requires custom.level")
        transpose = custom.get("transpose")
        return custom["level"], None if transpose is None else tuple(transpose)

    try:
        format, transpose = _PREDEFINED_LEVELS[format_name]
    except KeyError as error:
        raise ValueError(f"unknown Binsparse format {format_name!r}") from error
    return format, transpose


def _npy_from_tensor(tensor: CustomTensor) -> tuple[np.ndarray, np.ndarray, Any]:
    fill_value = tensor.fill_value if tensor.fill is True else 0
    values = _element_values(tensor.level)
    dtype = (
        np.result_type(values.dtype, np.asarray(fill_value).dtype)
        if values.size > 0
        else np.asarray(fill_value).dtype
    )
    dense = np.full(tensor.shape, fill_value, dtype=dtype)
    pattern = np.zeros(tensor.shape, dtype=bool)
    transpose = tensor.transpose or tuple(range(len(tensor.shape)))
    for stored_coord, value in binsparse_to_coo(tensor):
        coord = tuple(stored_coord[transpose.index(d)] for d in range(len(transpose)))
        dense[coord] = value
        pattern[coord] = True
    return dense, pattern, fill_value


def _element_values(level: BinsparseLevel | None) -> np.ndarray:
    match level:
        case ElementLevel(values):
            return values
        case DenseLevel(_, child) | SparseLevel(_, child, _, _):
            return _element_values(child)
        case _:
            raise TypeError(f"unsupported level type {type(level).__name__}")


__all__ = [
    "binsparse_to_binsparse_main",
    "binsparse_to_npy_main",
    "npy_to_binsparse_main",
]

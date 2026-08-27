"""Compliance-test executables which round-trip through optional frameworks."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .cli import _load_fill_value, _load_header, _npy_from_tensor, _tensor_from_npy
from .conversions import (
    from_numpy,
    from_scipy,
    from_sparse,
    to_numpy,
    to_scipy,
    to_sparse,
)
from .io import load_binsparse, save_binsparse
from .reformat import alias_to_custom
from .tensor import BinsparseTensor

Converter = tuple[Callable[[BinsparseTensor], Any], Callable[[Any], BinsparseTensor]]


def _converter(name: str) -> Converter:
    converters: dict[str, Converter] = {
        "numpy": (to_numpy, from_numpy),
        "scipy": (to_scipy, from_scipy),
        "sparse": (to_sparse, from_sparse),
    }
    try:
        return converters[name]
    except KeyError as error:
        raise ValueError(f"unknown framework {name!r}") from error


def _roundtrip(tensor: BinsparseTensor, framework: str) -> BinsparseTensor:
    to_framework, from_framework = _converter(framework)
    return from_framework(to_framework(tensor))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        return 2
    framework, operation, *paths = args
    if operation == "npy_to_binsparse" and len(paths) == 5:
        tensor_in, pattern_in, fill_in, header_in, tensor_out = paths
        header = _load_header(header_in)
        tensor = _tensor_from_npy(
            np.load(tensor_in, allow_pickle=False),
            np.asarray(np.load(pattern_in, allow_pickle=False), dtype=bool),
            _load_fill_value(fill_in),
            header,
        )
        save_binsparse(_roundtrip(tensor, framework), tensor_out, header=header)
        return 0
    if operation == "binsparse_to_npy" and len(paths) == 4:
        tensor_in, tensor_out, pattern_out, fill_out = paths
        tensor = alias_to_custom(_roundtrip(load_binsparse(tensor_in), framework))
        dense, pattern, fill = _npy_from_tensor(tensor)
        np.save(tensor_out, dense)
        np.save(pattern_out, pattern)
        np.save(fill_out, np.asarray(fill))
        return 0
    if operation == "binsparse_to_binsparse" and len(paths) == 2:
        tensor_in, tensor_out = paths
        save_binsparse(_roundtrip(load_binsparse(tensor_in), framework), tensor_out)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

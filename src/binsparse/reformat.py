"""Reformat Binsparse tensors through coordinate lists."""

from collections.abc import Iterable, Iterator
from itertools import groupby, product
from typing import Any

import numpy as np

from .container import InMemoryBinsparseContainer
from .tensor import (
    _PREDEFINED_LEVELS,
    BinsparseLevel,
    BinsparseTensor,
    CustomTensor,
    DenseLevel,
    ElementLevel,
    SparseLevel,
)

def alias_to_custom(tensor: BinsparseTensor) -> CustomTensor:
    """Return the custom-level representation of any Binsparse tensor."""
    container = InMemoryBinsparseContainer()
    tensor.serialize(container, copy=False, alias=False)
    custom = CustomTensor.parse(container, copy=False)
    assert isinstance(custom, CustomTensor)
    return custom

def custom_to_alias(
    tensor: BinsparseTensor, format_name: str | None = None
) -> BinsparseTensor:
    """Return the predefined alias for a custom tensor, when one exists."""
    container = InMemoryBinsparseContainer()
    tensor.serialize(container, copy=False, alias=True)
    alias = BinsparseTensor.parse(container, copy=False)
    if format_name is not None and alias.format != format_name:
        raise ValueError(f"custom layout is not compatible with {format_name!r}")
    return alias

def binsparse_to_coo(
    tensor: BinsparseTensor,
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Iterate over a tensor's explicitly stored coordinate/value pairs."""
    tensor = tensor if isinstance(tensor, CustomTensor) else alias_to_custom(tensor)
    shape = tensor.shape
    root = tensor.level
    assert root is not None

    def level_to_coo(
        level: BinsparseLevel, coords: list[tuple[int, ...]], shape: tuple[int, ...]
    ) -> Iterable[tuple[tuple[Any, ...], Any]]:
        match level:
            case ElementLevel(values):
                return zip(coords, values, strict=True)
            case DenseLevel(rank, child):
                coords = [
                    (*coord, *suffix)
                    for coord in coords
                    for suffix in product(*(range(size) for size in shape[:rank]))
                ]
                return level_to_coo(child, coords, shape[rank:])
            case SparseLevel(rank, child, indices, pointers):
                pointers = (
                    np.array([0, indices[0].size])
                    if pointers is None
                    else pointers
                )
                coords = [
                    (*coord, *(index[q] for index in indices))
                    for p, coord in enumerate(coords)
                    for q in range(pointers[p], pointers[p + 1])
                ]
                return level_to_coo(child, coords, shape[rank:])
            case _:
                raise TypeError(f"unsupported level type {type(level).__name__}")

    yield from level_to_coo(root, [()], shape)


def coo_to_binsparse(
    format: dict[str, Any],
    entries: Iterable[tuple[tuple[Any, ...], Any]],
    *,
    shape: tuple[int, ...],
) -> CustomTensor:
    """Build a custom tensor from a nested format descriptor and COO entries."""
    entries = sorted((tuple(coord), value) for coord, value in entries)
    for coord, _ in entries:
        if len(coord) != len(shape):
            raise ValueError("coordinate rank does not match tensor rank")
        if any(i < 0 or i >= n for i, n in zip(coord, shape, strict=True)):
            raise ValueError("coordinate is out of bounds")

    def coo_to_level(
        format: dict[str, Any],
        entries: list[tuple[tuple[Any, ...], Any]],
        ptr: np.ndarray,
        shape: tuple[int, ...],
        root: bool = False,
    ) -> BinsparseLevel:
        match format["level_desc"]:
            case "element":
                if np.any(np.diff(ptr) != 1):
                    raise ValueError("each element position must have one value")
                return ElementLevel(np.asarray([value for _, value in entries]))
            case "dense":
                rank = format["rank"]
                groups = [
                    [
                        (coord[rank:], value)
                        for coord, value in entries[ptr[p] : ptr[p + 1]]
                        if coord[:rank] == suffix
                    ]
                    for p in range(ptr.size - 1)
                    for suffix in product(*(range(size) for size in shape[:rank]))
                ]
                entries = [entry for group in groups for entry in group]
                ptr = np.cumsum([0, *(len(group) for group in groups)])
                return DenseLevel(
                    rank,
                    coo_to_level(format["level"], entries, ptr, shape[rank:]),
                )
            case "sparse":
                rank = format["rank"]
                grouped = [
                    [
                        (coord, list(group))
                        for coord, group in groupby(
                            entries[ptr[p] : ptr[p + 1]],
                            lambda entry: entry[0][:rank],
                        )
                    ]
                    for p in range(ptr.size - 1)
                ]
                groups = [
                    [(coord[rank:], value) for coord, value in group]
                    for fiber in grouped
                    for _, group in fiber
                ]
                entries = [entry for group in groups for entry in group]
                child_ptr = np.cumsum([0, *(len(group) for group in groups)])
                new_indices = tuple(
                    np.asarray([coord[d] for fiber in grouped for coord, _ in fiber])
                    for d in range(rank)
                )
                new_pointers = np.cumsum(
                    [0, *(len(fiber) for fiber in grouped)], dtype=np.uint64
                )
                return SparseLevel(
                    rank,
                    coo_to_level(
                        format["level"], entries, child_ptr, shape[rank:]
                    ),
                    new_indices,
                    None if root else new_pointers,
                )
            case _:
                raise ValueError(f"unknown level descriptor {format['level_desc']!r}")

    return CustomTensor(
        shape,
        len(entries),
        level=coo_to_level(
            format,
            entries,
            np.array([0, len(entries)], dtype=np.uint64),
            shape,
            root=True,
        ),
    )

def reformat(tensor: BinsparseTensor, header: dict[str, Any]) -> BinsparseTensor:
    """Reformat *tensor* according to a Binsparse format descriptor."""
    source = tensor if isinstance(tensor, CustomTensor) else alias_to_custom(tensor)
    format_name = header["format"]
    if format_name == "custom":
        custom = header["custom"]
        format = custom["level"]
        transpose = custom.get("transpose")
    else:
        try:
            format, transpose = _PREDEFINED_LEVELS[format_name]
        except KeyError as error:
            raise ValueError(f"unknown Binsparse format {format_name!r}") from error

    source_transpose = source.transpose or tuple(range(len(source.shape)))
    target_transpose = transpose or tuple(range(len(source.shape)))
    entries = [
        (
            tuple(
                coord[source_transpose.index(d)]
                for d in target_transpose
            ),
            value,
        )
        for coord, value in binsparse_to_coo(source)
    ]
    stored_shape = tuple(source.shape[d] for d in target_transpose)
    result = coo_to_binsparse(format, entries, shape=stored_shape)
    result.shape = source.shape
    result.transpose = None if transpose is None else tuple(transpose)
    result.fill = source.fill
    result.fill_value = source.fill_value
    return result if format_name == "custom" else custom_to_alias(result, format_name)


__all__ = [
    "alias_to_custom",
    "custom_to_alias",
    "binsparse_to_coo",
    "coo_to_binsparse",
    "reformat",
]

"""Path-based Binsparse file I/O."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .container import (
    HDF5BinsparseContainer,
    NPZBinsparseContainer,
    ZarrBinsparseContainer,
)
from .tensor import BinsparseTensor


Pathish = str | PathLike[str]


def _suffix(path: Pathish) -> str:
    return Path(path).suffix.lower()


def _unsupported(path: Pathish) -> ValueError:
    return ValueError(f"unsupported Binsparse container extension {Path(path).suffix!r}")


def load_binsparse(
    path: Pathish,
    *,
    alias: bool | None = None,
    copy: bool | None = None,
) -> BinsparseTensor:
    """Load a Binsparse tensor from *path*, dispatching by file extension."""
    match _suffix(path):
        case ".npz":
            with np.load(path, allow_pickle=False) as archive:
                return BinsparseTensor.parse(
                    NPZBinsparseContainer(archive),
                    alias=alias,
                    copy=copy,
                )
        case ".h5" | ".hdf5":
            import h5py

            with h5py.File(path, "r") as file:
                return BinsparseTensor.parse(
                    HDF5BinsparseContainer(file),
                    alias=alias,
                    copy=copy,
                )
        case ".zarr":
            zarr = _zarr()
            group = zarr.open_group(path, mode="r")
            return BinsparseTensor.parse(
                ZarrBinsparseContainer(group),
                alias=alias,
                copy=copy,
            )
        case _:
            raise _unsupported(path)


def save_binsparse(
    tensor: BinsparseTensor,
    path: Pathish,
    *,
    alias: bool | None = None,
    copy: bool | None = None,
    header: Mapping[str, Any] | None = None,
) -> None:
    """Write *tensor* to *path*, dispatching by file extension."""
    match _suffix(path):
        case ".npz":
            archive: dict[str, np.ndarray] = {}
            npz_container = NPZBinsparseContainer(archive)
            tensor.serialize(npz_container, alias=alias, copy=copy)
            if header is not None:
                _preserve_header(npz_container, header)
            savez: Any = np.savez
            savez(path, **archive)
        case ".h5" | ".hdf5":
            import h5py

            with h5py.File(path, "w") as file:
                hdf5_container = HDF5BinsparseContainer(file)
                tensor.serialize(
                    hdf5_container,
                    alias=alias,
                    copy=copy,
                )
                if header is not None:
                    _preserve_header(hdf5_container, header)
        case ".zarr":
            zarr = _zarr()
            group = zarr.open_group(path, mode="w")
            zarr_container = ZarrBinsparseContainer(group)
            tensor.serialize(
                zarr_container,
                alias=alias,
                copy=copy,
            )
            if header is not None:
                _preserve_header(zarr_container, header)
        case _:
            raise _unsupported(path)


def _preserve_header(container: Any, header: Mapping[str, Any]) -> None:
    merged = _merge_header(container.read_header(), header)
    data_types = merged.pop("data_types")
    container.data_types.clear()
    container.data_types.update(data_types)
    container.write_header(merged)


def _merge_header(
    generated: Mapping[str, Any],
    supplied: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(generated)
    merged.update(supplied)
    generated_data_types = generated.get("data_types")
    supplied_data_types = supplied.get("data_types")
    if isinstance(generated_data_types, dict):
        data_types = dict(generated_data_types)
    else:
        data_types = {}
    if isinstance(supplied_data_types, dict):
        data_types.update(supplied_data_types)
    merged["data_types"] = data_types
    return merged


def _zarr() -> Any:
    try:
        import zarr
    except ImportError as error:
        raise ImportError(
            "Zarr Binsparse I/O requires the 'zarr' package"
        ) from error
    return zarr


__all__ = ["load_binsparse", "save_binsparse"]

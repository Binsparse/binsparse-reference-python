"""Path-based Binsparse file I/O."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

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
) -> None:
    """Write *tensor* to *path*, dispatching by file extension."""
    match _suffix(path):
        case ".npz":
            archive: dict[str, np.ndarray] = {}
            tensor.serialize(NPZBinsparseContainer(archive), alias=alias, copy=copy)
            savez: Any = np.savez
            savez(path, **archive)
        case ".h5" | ".hdf5":
            import h5py

            with h5py.File(path, "w") as file:
                tensor.serialize(
                    HDF5BinsparseContainer(file),
                    alias=alias,
                    copy=copy,
                )
        case ".zarr":
            zarr = _zarr()
            group = zarr.open_group(path, mode="w")
            tensor.serialize(
                ZarrBinsparseContainer(group),
                alias=alias,
                copy=copy,
            )
        case _:
            raise _unsupported(path)


def _zarr() -> Any:
    try:
        import zarr
    except ImportError as error:
        raise ImportError(
            "Zarr Binsparse I/O requires the 'zarr' package"
        ) from error
    return zarr


__all__ = ["load_binsparse", "save_binsparse"]

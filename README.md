# Binsparse Python Reference Implementation

This library is a reference implementation of the [Binsparse Binary Sparse Format Specification](https://binsparse.org) written using Python.

Binsparse is a cross-platform, embeddable format for storing sparse matrices and
tensors. This library implements Binsparse bindings for
[NumPy NPZ](https://numpy.org/doc/stable/reference/generated/numpy.savez.html),
[HDF5](https://www.hdfgroup.org/solutions/hdf5/), and
[Zarr](https://zarr.dev/) containers, with conversion interfaces for
[NumPy](https://numpy.org/), [SciPy](https://scipy.org/),
[PyTorch](https://pytorch.org/), and
[PyData/Sparse](https://sparse.pydata.org/).

## Python Binsparse Interface

The primary interface consists of `load_binsparse` and `save_binsparse`. These
functions select a container from the path extension (`.npz`, `.h5`/`.hdf5`, or
`.zarr`) and read or write a `BinsparseTensor`:

```python
from binsparse import load_binsparse, save_binsparse

tensor = load_binsparse("input.h5")
save_binsparse(tensor, "output.npz")
```

Parsed tensors expose their logical `shape`, `number_of_stored_values`, optional
fill value, and the arrays that define their storage format. Concrete tensor
classes represent the predefined Binsparse formats, including dense vectors and
matrices, compressed sparse rows and columns, doubly compressed matrices, and
coordinate formats. `CustomTensor` and the level classes (`DenseLevel`,
`SparseLevel`, and `ElementLevel`) represent arbitrary level-based format
descriptors.

For applications that already manage their own storage objects, the lower-level
`BinsparseContainer` interface separates tensor parsing and serialization from
the physical container. Adapters are provided for NPZ mappings, HDF5 files,
Zarr groups, and in-memory descriptors and buffers. Use
`BinsparseTensor.parse(container)` to read through an adapter and
`tensor.serialize(container)` to write through one.

The `alias` option controls whether predefined format names such as `CSR` are
preserved or expanded to custom level descriptors. The `copy` option allows a
caller to request a copy, permit whichever representation is required, or
require a zero-copy operation when the source and destination support it.

Conversion helpers in `binsparse.conversions` convert between Binsparse tensors
and NumPy, SciPy, PyTorch, or PyData/Sparse objects. NumPy support is included by
default; the other adapters require their corresponding optional dependency.

## Source
The source code for `binsparse` is available on GitHub at [https://github.com/Binsparse/binsparse-reference-python](https://github.com/Binsparse/binsparse-reference-python)

## Installation

`binsparse` is available on PyPi, and can be installed with pip:
```bash
pip install binsparse
```

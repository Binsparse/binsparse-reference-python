import json

import numpy as np
import pytest

from binsparse import load_binsparse, save_binsparse
from binsparse.container import HDF5BinsparseContainer
from binsparse.tensor import CSRMatrix, CustomTensor, DVECVector


def _csr() -> CSRMatrix:
    return CSRMatrix(
        (3, 4),
        3,
        pointers_to_1=np.array([0, 1, 1, 3], dtype=np.uint64),
        indices_1=np.array([2, 0, 3], dtype=np.uint32),
        values=np.array([5, 6, 7], dtype=np.int16),
    )


def test_npz_file_round_trip_dispatches_by_extension(tmp_path) -> None:
    path = tmp_path / "tensor.npz"

    save_binsparse(_csr(), path, alias=False)
    with np.load(path, allow_pickle=False) as archive:
        document = json.loads(str(archive["binsparse"].item()))
    assert set(document) == {"binsparse"}
    assert document["binsparse"]["format"] == "custom"
    result = load_binsparse(path, alias=False)

    assert isinstance(result, CustomTensor)
    assert result.shape == (3, 4)
    assert result.number_of_stored_values == 3


def test_hdf5_file_round_trip_dispatches_by_extension(tmp_path) -> None:
    path = tmp_path / "tensor.h5"

    save_binsparse(_csr(), path)
    import h5py

    with h5py.File(path, "r") as file:
        document = json.loads(file.attrs["binsparse"])
    assert set(document) == {"binsparse"}
    assert document["binsparse"]["format"] == "CSR"
    result = load_binsparse(path)

    assert isinstance(result, CSRMatrix)
    np.testing.assert_array_equal(result.pointers_to_1, [0, 1, 1, 3])
    np.testing.assert_array_equal(result.indices_1, [2, 0, 3])
    np.testing.assert_array_equal(result.values, [5, 6, 7])


def test_hdf5_bint8_uses_plain_uint8_dataset(tmp_path) -> None:
    import h5py

    path = tmp_path / "boolean.h5"
    source = np.array([True, False, True], dtype=np.bool_)
    save_binsparse(DVECVector((3,), 3, values=source), path)

    with h5py.File(path, "r") as file:
        assert file["values"].dtype == np.dtype("uint8")
        assert file["values"].dtype.metadata is None
        document = json.loads(file.attrs["binsparse"])
        assert document["binsparse"]["data_types"]["values"] == "bint8"
        decoded = HDF5BinsparseContainer(file).read_buffer("values")
        assert decoded.dtype == np.dtype("bool")
        np.testing.assert_array_equal(decoded, source)

    result = load_binsparse(path)
    assert isinstance(result, DVECVector)
    assert result.values.dtype == np.dtype("bool")
    np.testing.assert_array_equal(result.values, source)


def test_file_io_rejects_unknown_extension(tmp_path) -> None:
    path = tmp_path / "tensor.bin"

    with pytest.raises(ValueError, match="unsupported.*extension"):
        save_binsparse(_csr(), path)
    with pytest.raises(ValueError, match="unsupported.*extension"):
        load_binsparse(path)

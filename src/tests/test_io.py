import numpy as np
import pytest

from binsparse import load_binsparse, save_binsparse
from binsparse.tensor import CSRMatrix, CustomTensor


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
    result = load_binsparse(path, alias=False)

    assert isinstance(result, CustomTensor)
    assert result.shape == (3, 4)
    assert result.number_of_stored_values == 3


def test_hdf5_file_round_trip_dispatches_by_extension(tmp_path) -> None:
    path = tmp_path / "tensor.h5"

    save_binsparse(_csr(), path)
    result = load_binsparse(path)

    assert isinstance(result, CSRMatrix)
    np.testing.assert_array_equal(result.pointers_to_1, [0, 1, 1, 3])
    np.testing.assert_array_equal(result.indices_1, [2, 0, 3])
    np.testing.assert_array_equal(result.values, [5, 6, 7])


def test_file_io_rejects_unknown_extension(tmp_path) -> None:
    path = tmp_path / "tensor.bin"

    with pytest.raises(ValueError, match="unsupported.*extension"):
        save_binsparse(_csr(), path)
    with pytest.raises(ValueError, match="unsupported.*extension"):
        load_binsparse(path)

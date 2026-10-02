"""PyData/Sparse conversion tests, skipped when the optional extra is absent."""

import numpy as np
import pytest

from binsparse.conversions import from_sparse, to_sparse
from binsparse.tensor import COORMatrix, CustomTensor, ElementLevel, SparseLevel

sparse = pytest.importorskip("sparse")


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("copy", [None, True, False])
@pytest.mark.parametrize("duplicates", [False, True])
def test_pydata_sparse_is_sorted_and_unique(ndim, copy, duplicates) -> None:
    coordinates = [2, 0, 2, 1] if duplicates else [2, 0, 1]
    coords = np.array([coordinates] * ndim)
    data = np.arange(1, len(coordinates) + 1, dtype=np.float32)
    # Bypass the constructor's normalization to exercise noncanonical buffers.
    source = sparse.COO(
        coords.copy(),
        data.copy(),
        shape=(3,) * ndim,
        sorted=True,
        has_duplicates=False,
        fill_value=7,
    )
    if copy is False:
        with pytest.raises(ValueError, match="canonicalize"):
            from_sparse(source, copy=copy)
    else:
        tensor = from_sparse(source, copy=copy)
        result = to_sparse(tensor)
        assert tensor.number_of_stored_values == 3
        assert tensor.fill_value == 7
        assert result.dtype == data.dtype
        np.testing.assert_array_equal(result.coords, [[0, 1, 2]] * ndim)
        np.testing.assert_array_equal(
            result.data, [2, 4, 4] if duplicates else [2, 3, 1]
        )
    np.testing.assert_array_equal(source.coords, coords)
    np.testing.assert_array_equal(source.data, data)


def test_pydata_sparse_round_trip() -> None:
    source = sparse.COO.from_numpy(np.array([[0, 2, 0], [3, 0, 4]], dtype=np.float32))
    tensor = from_sparse(source)
    result = to_sparse(tensor)

    assert isinstance(tensor, COORMatrix)
    np.testing.assert_array_equal(result.todense(), source.todense())
    assert result.dtype == source.dtype


def test_pydata_sparse_copy_policy() -> None:
    source = sparse.COO.from_numpy(np.eye(2))
    shared = from_sparse(source, copy=False)
    copied = from_sparse(source, copy=True)

    assert isinstance(shared, COORMatrix)
    assert isinstance(copied, COORMatrix)
    assert np.shares_memory(shared.values, source.data)
    assert np.shares_memory(shared.indices_0, source.coords)
    assert not np.shares_memory(copied.values, source.data)
    assert not np.shares_memory(copied.indices_0, source.coords)

    with pytest.raises(ValueError, match="Binsparse.*single coordinate matrix"):
        to_sparse(shared, copy=False)

    shared_result = to_sparse(shared)
    copied_result = to_sparse(shared, copy=True)
    assert np.shares_memory(shared_result.data, shared.values)
    assert not np.shares_memory(shared_result.coords, source.coords)
    assert not np.shares_memory(copied_result.data, shared.values)


def test_pydata_sparse_nd_round_trip() -> None:
    source = sparse.COO.from_numpy(
        np.array(
            [
                [[0, 1], [2, 0]],
                [[3, 0], [0, 4]],
            ]
        )
    )
    tensor = from_sparse(source, copy=False)
    result = to_sparse(tensor)

    assert isinstance(tensor, CustomTensor)
    assert isinstance(tensor.level, SparseLevel)
    assert tensor.level.rank == 3
    assert isinstance(tensor.level.level, ElementLevel)
    assert np.shares_memory(tensor.level.level.values, source.data)
    assert all(np.shares_memory(index, source.coords) for index in tensor.level.indices)
    assert not np.shares_memory(result.coords, source.coords)
    assert np.shares_memory(result.data, source.data)
    np.testing.assert_array_equal(result.todense(), source.todense())

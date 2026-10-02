"""PyTorch conversion tests, skipped when the optional extra is absent."""

import numpy as np
import pytest

from binsparse.conversions import from_torch, to_torch
from binsparse.tensor import (
    COORMatrix,
    CSCMatrix,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
    SparseLevel,
)

torch = pytest.importorskip("torch")


def test_torch_dense_round_trip() -> None:
    source = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    tensor = from_torch(source)
    result = to_torch(tensor)

    assert isinstance(tensor, DMATRMatrix)
    np.testing.assert_array_equal(result.numpy(), source.numpy())


def test_torch_nd_dense_round_trip() -> None:
    source = torch.arange(24).reshape(2, 3, 4)
    tensor = from_torch(source, copy=False)
    result = to_torch(tensor, copy=False)

    assert isinstance(tensor, CustomTensor)
    assert isinstance(tensor.level, DenseLevel)
    assert tensor.level.rank == source.ndim
    assert result.data_ptr() == source.data_ptr()
    assert torch.equal(result, source)


def test_torch_csr_round_trip() -> None:
    source = torch.tensor([[0.0, 2.0], [3.0, 0.0]]).to_sparse_csr()
    tensor = from_torch(source)
    result = to_torch(tensor)

    assert isinstance(tensor, CSRMatrix)
    assert torch.equal(result.to_dense(), source.to_dense())


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("copy", [None, True, False])
def test_torch_uncoalesced_coo_is_sorted_and_unique(ndim, copy) -> None:
    coords = torch.tensor([[2, 0, 2, 1]] * ndim)
    values = torch.tensor([1.0, 2.0, -1.0, 4.0])
    source = torch.sparse_coo_tensor(
        coords.clone(),
        values.clone(),
        (3,) * ndim,
    )
    if copy is False:
        with pytest.raises(ValueError, match="canonicalize"):
            from_torch(source, copy=copy)
    else:
        tensor = from_torch(source, copy=copy)
        result = to_torch(tensor)
        assert tensor.number_of_stored_values == 3
        assert result.dtype == source.dtype
        assert torch.equal(result._indices(), torch.tensor([[0, 1, 2]] * ndim))
        assert torch.equal(result._values(), torch.tensor([2.0, 4.0, 0.0]))
        assert torch.equal(result.to_dense(), source.to_dense())
    assert not source.is_coalesced()
    assert torch.equal(source._indices(), coords)
    assert torch.equal(source._values(), values)


@pytest.mark.parametrize("layout", ["coo", "csr", "csc"])
@pytest.mark.parametrize("empty", [False, True])
def test_torch_canonical_sparse_copy_policy(layout, empty) -> None:
    dense = torch.zeros((3, 3)) if empty else torch.eye(3)
    source = dense.to_sparse(layout=getattr(torch, f"sparse_{layout}"))
    shared = from_torch(source, copy=False)
    copied = from_torch(source, copy=True)
    assert isinstance(shared, (COORMatrix, CSRMatrix, CSCMatrix))
    assert isinstance(copied, (COORMatrix, CSRMatrix, CSCMatrix))
    original_values = source.values().numpy()
    np.testing.assert_array_equal(copied.values, original_values)
    if not empty:
        assert np.shares_memory(shared.values, original_values)
        assert not np.shares_memory(copied.values, original_values)
        if layout == "coo":
            assert isinstance(shared, COORMatrix)
            assert isinstance(copied, COORMatrix)
            assert np.shares_memory(shared.indices_0, source.indices().numpy())
            assert not np.shares_memory(copied.indices_0, source.indices().numpy())
        else:
            indices = source.col_indices() if layout == "csr" else source.row_indices()
            assert np.shares_memory(shared.indices_1, indices.numpy())
            assert not np.shares_memory(copied.indices_1, indices.numpy())


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
def test_torch_nd_coo_round_trip(shape) -> None:
    coordinates = torch.tensor([[0, size - 1] for size in shape])
    source = torch.sparse_coo_tensor(
        coordinates, torch.tensor([2.0, 3.0]), shape
    ).coalesce()
    tensor = from_torch(source, copy=False)
    result = to_torch(tensor)

    assert isinstance(tensor, CustomTensor)
    assert isinstance(tensor.level, SparseLevel)
    assert tensor.level.rank == len(shape)
    assert isinstance(tensor.level.level, ElementLevel)
    assert all(
        index.__array_interface__["data"][0] == source._indices()[dimension].data_ptr()
        for dimension, index in enumerate(tensor.level.indices)
    )
    assert torch.equal(result.to_dense(), source.to_dense())
    with pytest.raises(ValueError, match="combine"):
        to_torch(tensor, copy=False)


def test_torch_copy_policy() -> None:
    source = torch.tensor([1.0, 2.0])
    shared = from_torch(source, copy=False)
    copied = from_torch(source, copy=True)

    assert isinstance(shared, DVECVector)
    assert isinstance(copied, DVECVector)
    assert shared.values.__array_interface__["data"][0] == source.data_ptr()
    assert copied.values.__array_interface__["data"][0] != source.data_ptr()

    shared_result = to_torch(shared, copy=False)
    copied_result = to_torch(shared, copy=True)
    assert shared_result.data_ptr() == shared.values.__array_interface__["data"][0]
    assert copied_result.data_ptr() != shared.values.__array_interface__["data"][0]


def test_torch_copy_true_handles_tensors_requiring_grad() -> None:
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    copied = from_torch(source, copy=True)

    assert isinstance(copied, DVECVector)
    assert copied.values.__array_interface__["data"][0] != source.data_ptr()
    np.testing.assert_array_equal(copied.values, source.detach().numpy())

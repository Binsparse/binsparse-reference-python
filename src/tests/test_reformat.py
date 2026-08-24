import numpy as np
import pytest

from binsparse.reformat import (
    alias_to_custom,
    binsparse_to_coo,
    coo_to_binsparse,
    reformat,
)
from binsparse.tensor import CSCMatrix, CSRMatrix, CustomTensor, DenseLevel, SparseLevel


CSR_FORMAT = {
    "level_desc": "dense",
    "rank": 1,
    "level": {
        "level_desc": "sparse",
        "rank": 1,
        "level": {"level_desc": "element"},
    },
}


def test_binsparse_to_coo_iterates_csr_entries() -> None:
    tensor = CSRMatrix(
        (3, 4),
        3,
        pointers_to_1=np.array([0, 1, 1, 3], dtype=np.uint64),
        indices_1=np.array([2, 0, 3], dtype=np.uint32),
        values=np.array([5, 6, 7], dtype=np.int16),
    )

    entries = list(binsparse_to_coo(tensor))
    assert entries == [
        ((0, 2), 5),
        ((2, 0), 6),
        ((2, 3), 7),
    ]
    assert all(isinstance(coord[1], np.uint32) for coord, _ in entries)


def test_coo_round_trip_rebuilds_custom_levels() -> None:
    tensor = CSRMatrix(
        (3, 4),
        3,
        pointers_to_1=np.array([0, 1, 1, 3], dtype=np.uint64),
        indices_1=np.array([2, 0, 3], dtype=np.uint32),
        values=np.array([5, 6, 7], dtype=np.int16),
    )
    custom = alias_to_custom(tensor)
    entries = list(binsparse_to_coo(custom))

    rebuilt = coo_to_binsparse(CSR_FORMAT, reversed(entries), shape=custom.shape)

    assert isinstance(rebuilt, CustomTensor)
    assert rebuilt.number_of_stored_values == 3
    assert isinstance(rebuilt.level, DenseLevel)
    assert isinstance(rebuilt.level.level, SparseLevel)
    assert rebuilt.level.level.pointers_to_next is not None
    assert rebuilt.level.level.pointers_to_next.dtype == np.uint64
    assert list(binsparse_to_coo(rebuilt)) == entries


def test_coo_to_binsparse_rejects_duplicate_coordinates() -> None:
    with pytest.raises(ValueError, match="one value"):
        coo_to_binsparse(
            CSR_FORMAT, [((0, 0), 1), ((0, 0), 2)], shape=(1, 1)
        )


def test_reformat_changes_predefined_layout() -> None:
    tensor = CSRMatrix(
        (3, 4),
        3,
        fill=True,
        fill_value=-1,
        pointers_to_1=np.array([0, 1, 1, 3], dtype=np.uint64),
        indices_1=np.array([2, 0, 3], dtype=np.uint32),
        values=np.array([5, 6, 7], dtype=np.int16),
    )

    result = reformat(tensor, {"format": "CSC"})

    assert isinstance(result, CSCMatrix)
    assert result.shape == tensor.shape
    assert result.fill is True
    assert result.fill_value == -1
    np.testing.assert_array_equal(result.pointers_to_1, [0, 1, 1, 2, 3])
    np.testing.assert_array_equal(result.indices_1, [2, 0, 2])
    np.testing.assert_array_equal(result.values, [6, 5, 7])


def test_reformat_accepts_custom_descriptor() -> None:
    tensor = CSRMatrix(
        (2, 3),
        2,
        pointers_to_1=np.array([0, 1, 2], dtype=np.uint64),
        indices_1=np.array([2, 0], dtype=np.uint16),
        values=np.array([4, 9], dtype=np.int8),
    )
    format = {
        "level_desc": "sparse",
        "rank": 2,
        "level": {"level_desc": "element"},
    }

    result = reformat(
        tensor, {"format": "custom", "custom": {"level": format}}
    )

    assert isinstance(result, CustomTensor)
    assert result.transpose is None
    assert list(binsparse_to_coo(result)) == [((0, 2), 4), ((1, 0), 9)]

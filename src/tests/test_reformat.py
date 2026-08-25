import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "bin"))

from reformat import (
    binsparse_to_coo,
    coo_to_binsparse,
    reformat,
)
from binsparse.tensor import (
    CSCMatrix,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    ElementLevel,
    SparseLevel,
)


CSR_FORMAT = {
    "level_desc": "dense",
    "rank": 1,
    "level": {
        "level_desc": "sparse",
        "rank": 1,
        "level": {"level_desc": "element"},
    },
}


def _custom_csr() -> CustomTensor:
    return CustomTensor(
        (3, 4),
        3,
        level=DenseLevel(
            1,
            SparseLevel(
                1,
                ElementLevel(np.array([5, 6, 7], dtype=np.int16)),
                (np.array([2, 0, 3], dtype=np.uint32),),
                np.array([0, 1, 1, 3], dtype=np.uint64),
            ),
        ),
    )


def test_binsparse_to_coo_iterates_csr_entries() -> None:
    entries = list(binsparse_to_coo(_custom_csr()))
    assert entries == [
        ((0, 2), 5),
        ((2, 0), 6),
        ((2, 3), 7),
    ]
    assert all(isinstance(coord[1], np.uint32) for coord, _ in entries)


def test_coo_round_trip_rebuilds_custom_levels() -> None:
    custom = _custom_csr()
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


def test_reformat_applies_requested_data_types() -> None:
    tensor = CSRMatrix(
        (3, 4),
        3,
        fill=True,
        fill_value=-1,
        pointers_to_1=np.array([0, 1, 1, 3], dtype=np.uint64),
        indices_1=np.array([2, 0, 3], dtype=np.uint64),
        values=np.array([5, 6, 7], dtype=np.int64),
    )

    result = reformat(
        tensor,
        {
            "format": "CSC",
            "data_types": {
                "pointers_to_1": "uint32",
                "indices_1": "uint16",
                "values": "float32",
                "fill_value": "int16",
            },
        },
    )

    assert isinstance(result, CSCMatrix)
    assert result.pointers_to_1.dtype == np.uint32
    assert result.indices_1.dtype == np.uint16
    assert result.values.dtype == np.float32
    assert isinstance(result.fill_value, np.int16)


def test_reformat_applies_iso_data_type() -> None:
    tensor = CSRMatrix(
        (3, 4),
        3,
        pointers_to_1=np.array([0, 1, 1, 3], dtype=np.uint64),
        indices_1=np.array([2, 0, 3], dtype=np.uint64),
        values=np.array([5, 5, 5], dtype=np.int64),
    )

    result = reformat(
        tensor,
        {"format": "CSR", "data_types": {"values": "iso[int16]"}},
    )

    assert isinstance(result, CSRMatrix)
    assert result.values.dtype == np.int16
    assert result.values.strides == (0,)
    np.testing.assert_array_equal(result.values, [5, 5, 5])


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

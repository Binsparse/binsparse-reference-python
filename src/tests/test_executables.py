import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from binsparse import load_binsparse
from binsparse.tensor import CSRMatrix


BIN = Path(__file__).parents[2] / "bin"


def test_binsparse_tests_executables_round_trip_npz(tmp_path) -> None:
    dense = np.full((3, 4), -1, dtype=np.int16)
    dense[0, 2] = 5
    dense[2, 0] = 6
    dense[2, 3] = 7
    pattern = dense != -1
    fill_value = np.array(-1, dtype=np.int16)
    header = {
        "format": "CSR",
        "data_types": {
            "pointers_to_1": "uint64",
            "indices_1": "uint32",
            "values": "int16",
            "fill_value": "int16",
        },
    }

    tensor_in = tmp_path / "tensor.npy"
    pattern_in = tmp_path / "pattern.npy"
    fill_value_in = tmp_path / "fill_value.npy"
    header_in = tmp_path / "header.json"
    binsparse_out = tmp_path / "tensor.npz"
    dense_out = tmp_path / "dense_out.npy"
    pattern_out = tmp_path / "pattern_out.npy"
    fill_value_out = tmp_path / "fill_value_out.npy"
    roundtrip_out = tmp_path / "roundtrip.npz"

    np.save(tensor_in, dense)
    np.save(pattern_in, pattern)
    np.save(fill_value_in, fill_value)
    header_in.write_text(json.dumps(header))

    for executable in (
        "npy_to_binsparse",
        "binsparse_to_npy",
        "binsparse_to_binsparse",
    ):
        assert os.access(BIN / executable, os.X_OK)

    subprocess.run(
        [
            sys.executable,
            BIN / "npy_to_binsparse",
            tensor_in,
            pattern_in,
            fill_value_in,
            header_in,
            binsparse_out,
        ],
        check=True,
    )

    tensor = load_binsparse(binsparse_out)
    assert isinstance(tensor, CSRMatrix)
    assert tensor.fill_value == np.int16(-1)
    np.testing.assert_array_equal(tensor.pointers_to_1, [0, 1, 1, 3])
    np.testing.assert_array_equal(tensor.indices_1, [2, 0, 3])
    np.testing.assert_array_equal(tensor.values, [5, 6, 7])

    subprocess.run(
        [
            sys.executable,
            BIN / "binsparse_to_npy",
            binsparse_out,
            dense_out,
            pattern_out,
            fill_value_out,
        ],
        check=True,
    )

    np.testing.assert_array_equal(np.load(dense_out), dense)
    np.testing.assert_array_equal(np.load(pattern_out), pattern)
    np.testing.assert_array_equal(np.load(fill_value_out), fill_value)

    subprocess.run(
        [
            sys.executable,
            BIN / "binsparse_to_binsparse",
            binsparse_out,
            roundtrip_out,
        ],
        check=True,
    )

    roundtrip = load_binsparse(roundtrip_out)
    assert isinstance(roundtrip, CSRMatrix)
    np.testing.assert_array_equal(roundtrip.pointers_to_1, tensor.pointers_to_1)
    np.testing.assert_array_equal(roundtrip.indices_1, tensor.indices_1)
    np.testing.assert_array_equal(roundtrip.values, tensor.values)

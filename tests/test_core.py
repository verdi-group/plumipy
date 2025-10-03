import numpy as np
import pytest

from plumipy.core import Photoluminescence


@pytest.fixture
def photoluminescence():
    return Photoluminescence()


@pytest.mark.parametrize(
    ("iv_low", "iv_high", "result"),
    [
        (0, 5, np.arange(0, 5)),
        (0, 10, np.arange(0, 10)),
    ],
)
def test_get_direct_mesh(photoluminescence, iv_low, iv_high, result):
    assert all(photoluminescence.get_direct_mesh(iv_low, iv_high, np.pi) == result)

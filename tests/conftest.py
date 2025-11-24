"""Project-wide pytest fixtures & hooks.

Docs: https://docs.pytest.org/en/stable/how-to/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import pytest
import numpy as np

from pathlib import Path


@pytest.fixture
def files_path():
    """Path to the data files used for the tests."""
    return Path(__file__).parent / "files"


@pytest.fixture()
def custom_serializer():
    def get_equally_spaced(array, max_number=None):
        """Get equally spaced number of elements from an array."""
        if max_number is None or max_number > len(array):
            return array

        indices = np.linspace(0, len(array) - 1, max_number, dtype=int)

        if isinstance(array, np.ndarray):
            return array[indices]
        if isinstance(array, (list, tuple)):
            return [array[i] for i in indices]

    def factory(item, max_number=None):
        if isinstance(item, (str, bool)):
            return item
        if isinstance(item, (list, tuple)):
            return [
                factory(el, max_number) for el in get_equally_spaced(item, max_number)
            ]
        if isinstance(item, dict):
            return {k: factory(v, max_number) for k, v in item.items()}
        if isinstance(item, (float, int)):
            return round(float(item), 5)
        if isinstance(item, np.ndarray):
            if np.iscomplexobj(item):
                return [factory(item.real, max_number), factory(item.imag, max_number)]
            else:
                return factory(
                    get_equally_spaced(item, max_number).tolist(), max_number
                )

        return item

    return factory

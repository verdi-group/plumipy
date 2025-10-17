import numpy as np
import pytest

from plumipy.core import Photoluminescence, calculate_spectrum
from pathlib import Path


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


def test_calculate_spectrum(data_regression):
    fixtures_directory = Path(__file__).parent / "fixtures"
    gs_structure_path = fixtures_directory / "CONTCAR_GS"
    es_structure_path = fixtures_directory / "CONTCAR_ES"
    phonons_band_path = fixtures_directory / "band.yaml"

    spectrum_data = calculate_spectrum(
        gs_structure_path=gs_structure_path,
        es_structure_path=es_structure_path,
        phonon_band_path=phonons_band_path,
    )
    spectrum_data[0]
    data_regression.check({"positions_gs": spectrum_data[0].tolist()})

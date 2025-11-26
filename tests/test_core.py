import numpy as np
import pytest

from plumipy.core import Photoluminescence, calculate_spectrum


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


def test_calculate_spectrum(files_path, data_regression, custom_serializer):
    """Integration test of the `calculate_spectrum` function."""

    (
        positions_gs,
        positions_es,
        qk,
        (a, partial_hr),
        (energy_mev_positive, specfun_energy),
        (t_fs, s_t, s_t_exact),
        (g_t),
        (energy_mev, a_e),
        (l_e),
        ipr,
    ) = calculate_spectrum(
        es_structure_path=files_path / "Si_CONTCAR_ES",
        gs_structure_path=files_path / "Si_CONTCAR_GS",
        phonon_band_path=files_path / "Si_band.yaml",
    )
    data_regression.check(
        custom_serializer(
            {
                "positions_gs": positions_gs,
                "positions_es": positions_es,
                "qk": qk,
                "a": a,
                "partial_hr": partial_hr,
                "energy_mev_positive": energy_mev_positive,
                "specfun_energy": specfun_energy,
                "t_fs": t_fs,
                "s_t": s_t,
                "s_t_exact": s_t_exact,
                "g_t": g_t,
                "energy_mev": energy_mev,
                "a_e": a_e,
                "l_e": l_e,
                "ipr": ipr,
            },
            max_number=5,
        )
    )

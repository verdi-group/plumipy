import pytest
from plumipy.parsers import ReadFiles


def test_read_structure(files_path, data_regression, custom_serializer):
    atomic_positions, atoms = ReadFiles().read_structure(files_path / "Si_CONTCAR_ES")
    data_regression.check(
        custom_serializer(
            {
                "atomic_positions": atomic_positions,
                "atoms": atoms,
            }
        )
    )


@pytest.mark.parametrize(
    "test_file",
    [
        "Si_band.yaml",
        "parsers/phonopy/qpoints.yaml",
    ],
)
def test_read_phonons_phonopy(
    files_path, data_regression, custom_serializer, test_file
):
    atomic_masses, freqs, eigenvectors = ReadFiles().read_phonons_phonopy(
        files_path / test_file
    )
    data_regression.check(
        custom_serializer(
            {
                "atomic_masses": atomic_masses,
                "freqs_keys": list(freqs.keys()),
                "freqs_values": list(freqs.values()),
                "eigenvectors_keys": list(eigenvectors.keys()),
                "eigenvectors_real": [val.real for val in eigenvectors.values()],
                "eigenvectors_imag": [val.imag for val in eigenvectors.values()],
            },
            max_number=5,
        )
    )

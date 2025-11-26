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


def test_read_phonons_phonopy(files_path, data_regression, custom_serializer):
    atomic_masses, freqs, normal_modes = ReadFiles().read_phonons_phonopy(
        files_path / "Si_band.yaml"
    )
    data_regression.check(
        custom_serializer(
            {
                "atomic_masses": atomic_masses,
                "freqs": freqs,
                "normal_modes": normal_modes,
            },
            max_number=5,
        )
    )

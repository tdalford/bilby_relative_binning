import pycentricity.waveform as wf


def test_generate_compliant_waveform():
    """
    Attempt to generate a waveform that has dimensionless component spins within -1 and 0.6
    """
    sampling_frequency = 4096.
    minimum_frequency = 10.

    good_parameters = dict(
        mass_1=30,
        mass_2=35,
        eccentricity=0.1,
        luminosity_distance=440.0,
        theta_jn=0.4,
        psi=0.1,
        phase=1.2,
        geocent_time=0.0,
        ra=45,
        dec=5.73,
        chi_1=0.4,
        chi_2=0.5,
    )

    seobnre_waveform = wf.seobnre_bbh_with_spin_and_eccentricity(
        good_parameters, sampling_frequency, minimum_frequency
    )
    assert type(seobnre_waveform['plus']) is list
    assert type(seobnre_waveform['cross']) is list


def test_generate_noncompliant_waveform():
    """
    Attempt to generate a waveform that has dimensionless component spins greater than 0.6
    """
    sampling_frequency = 4096.
    minimum_frequency = 10.

    bad_parameters = dict(
        mass_1=30,
        mass_2=35,
        eccentricity=0.1,
        luminosity_distance=440.0,
        theta_jn=0.4,
        psi=0.1,
        phase=1.2,
        geocent_time=0.0,
        ra=45,
        dec=5.73,
        chi_1=0.7,
        chi_2=0.8,
    )

    seobnre_waveform = wf.seobnre_bbh_with_spin_and_eccentricity(
        bad_parameters, sampling_frequency, minimum_frequency
    )
    assert len(seobnre_waveform['plus']) == 0
    assert len(seobnre_waveform['cross']) == 0



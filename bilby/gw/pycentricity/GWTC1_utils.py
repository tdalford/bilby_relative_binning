"""

A file containing general utility functions relating to GWTC-1 for the pycentricity package.

"""
parameter_keys = dict(
    chirp_mass='$\mathcal{M}$ [M$_{\odot}$]',
    mass_ratio='$q$',
    mass_1='$m_1$',
    mass_2='$m_2$',
    luminosity_distance='$d_\mathrm{L}$ [Mpc]',
    ra='RA',
    dec='DEC',
    chi_eff='$\chi_\mathrm{eff}$',
    chi_1='$\chi_1$',
    chi_2='$\chi_2$',
    theta_jn='$\\theta_\mathrm{jn}$',
    phase='$\phi$',
    psi='$\psi$',
    log_eccentricity='log$_{10}(e)$',
    geocent_time='$t_\mathrm{geo}$ [s]'
)
search_keys = [
    'chirp_mass',
    'mass_ratio',
    'luminosity_distance',
    'ra',
    'dec',
    'chi_1',
    'chi_2',
    'theta_jn',
    'phase',
    'psi',
    'geocent_time'
]

trigger_time = dict(
    GW150914=1126259462.391,
    GW151012=1128678900.4,
    GW151226=1135136350.6,
    GW170104=1167559936.6,
    GW170608=1180922494.5,
    GW170729=1185389807.3,
    GW170809=1186302519.8,
    GW170814=1186741861.5,
    GW170818=1187058327.1,
    GW170823=1187529256.5,
)

event_channels = dict(
    GW150914=dict(H1="DCS-CALIB_STRAIN_C02", L1="DCS-CALIB_STRAIN_C02"),
    GW151012=dict(H1="DCS-CALIB_STRAIN_C02", L1="DCS-CALIB_STRAIN_C02"),
    GW151226=dict(H1="DCS-CALIB_STRAIN_C02", L1="DCS-CALIB_STRAIN_C02"),
    GW170104=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
    GW170608=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
    GW170729=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
    GW170809=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
    GW170814=dict(
        H1="DCH-CLEAN_STRAIN_C02",
        L1="DCH-CLEAN_STRAIN_C02",
        V1="Hrec_hoft_V1O2Repro2A_16384Hz",
    ),
    GW170818=dict(
        H1="DCH-CLEAN_STRAIN_C02",
        L1="DCH-CLEAN_STRAIN_C02",
        V1="Hrec_hoft_V1O2Repro2A_16384Hz",
    ),
    GW170823=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
)

event_detectors = {
    key: list(event_channels[key].keys()) for key in event_channels.keys()
}

event_duration = dict(
    GW150914=8,
    GW151012=8,
    GW151226=8,
    GW170104=4,
    GW170608=16,
    GW170729=4,
    GW170809=4,
    GW170814=4,
    GW170818=4,
    GW170823=4,
)

minimum_frequency = dict(
    GW150914=20,
    GW151012=20,
    GW151226=20,
    GW170104=20,
    GW170608=30,
    GW170729=20,
    GW170809=20,
    GW170814=20,
    GW170818=20,
    GW170823=20,
)

sampling_frequency = dict(
    GW150914=4096,
    GW151012=4096,
    GW151226=4096,
    GW170104=4096,
    GW170608=4096,
    GW170729=4096,
    GW170809=4096,
    GW170814=4096,
    GW170818=4096,
    GW170823=4096,
)

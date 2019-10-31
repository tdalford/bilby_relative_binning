#!/usr/bin/env python
"""
"""
import numpy as np
import bilby
from bilby.core.prior import Uniform, Cosine

duration = 4.
sampling_frequency = 2048.

outdir = 'outdir'
label = 'ptemcee'

np.random.seed(88170235)

injection_parameters = dict(
    chirp_mass=36., mass_ratio=0.6, chi_1=0.0, chi_2=0.0,
    luminosity_distance=2000., cos_theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_arguments = dict(waveform_approximant='IMRPhenomD',
                          reference_frequency=100.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)
for ifo in ifos:
    ifo.minimum_frequency = 20
    ifo.maximum_frequency = 500

priors = bilby.gw.prior.PriorDict()
priors['chirp_mass'] = Uniform(35.0, 37.0, 'chirp_mass')
priors['mass_ratio'] = Uniform(0.5, 1, 'mass_ratio')
priors['chi_1'] = 0
priors['chi_2'] = 0
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')
priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(
    name='luminosity_distance', minimum=1e2, maximum=5e3, unit='Mpc', boundary=None)
priors['dec'] = Cosine(name='dec', boundary='reflective')
priors['ra'] = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors['cos_theta_jn'] = Uniform(name='cos_theta_jn', minimum=-1, maximum=1, boundary=None)
priors['psi'] = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
priors['phase'] = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')

for key in ['psi', 'cos_theta_jn']:
    priors[key] = injection_parameters[key]

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator,
    priors=priors, distance_marginalization=True, phase_marginalization=True,
    time_marginalization=True, jitter_time=False)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.core.sampler.run_sampler(
    likelihood=likelihood, priors=priors, sampler='ptemcee', ntemps=2,
    nwalkers=100, n_effective=500, iterations=5000, nburn=None,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    outdir=outdir, label=label)

# Make a corner plot.
result.plot_walkers()
result.plot_corner(['geocent_time', 'phase', 'luminosity_distance', 'chirp_mass', 'mass_ratio'])

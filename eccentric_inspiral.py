#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected eccentric binary black hole signal with masses & distnace
similar to GW150914.

This uses the same binary parameters that were used to make Figures 1, 2 & 5 in
Lower et al. (2018) -> arXiv:1806.05350.

For a more comprehensive look at what goes on in each step, refer to the
"basic_tutorial.py" example.
"""
from __future__ import division

import numpy as np
import bilby

duration = 4
sampling_frequency = 1024

outdir = 'outdir_eccentric_new'
label = 'eccentric_GW140914'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.
np.random.seed(150914)

injection_parameters = dict(
    mass_1=35., mass_2=30., eccentricity=0.1, luminosity_distance=440., chi_1=0.0, chi_2=0.0,
    theta_jn=0.4, psi=0.1, phase=1.2, geocent_time=1180002601.0, ra=45, dec=5.73)

waveform_arguments = dict(waveform_approximant='SEOBNRE',
                          sampling_frequency=1024, minimum_frequency=10)

# Create the waveform_generator using the LAL eccentric black hole no spins
# source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    time_domain_source_model=bilby.gw.source.lal_eccentric_binary_black_hole_with_spins,
    parameters=injection_parameters, waveform_arguments=waveform_arguments)

print("wf length time domain: " + str(len(waveform_generator.time_domain_strain(injection_parameters)['plus'])))
print("wf length freq domain: " + str(len(waveform_generator.frequency_domain_strain(injection_parameters)['plus'])))
# Setting up three interferometers (LIGO-Hanford (H1), LIGO-Livingston (L1), and
# Virgo (V1)) at their design sensitivities. The maximum frequency is set just
# prior to the point at which the waveform model terminates. This is to avoid
# any biases introduced from using a sharply terminating waveform model.
minimum_frequency = 10
maximum_frequency = 512

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = maximum_frequency
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

print("ifo length time domain: " + str(len(ifos[0].time_domain_strain)))
print("ifo length freq domain: " + str(len(ifos[0].frequency_domain_strain)))


# Now we set up the priors on each of the binary parameters.
priors = bilby.core.prior.PriorDict()
priors["mass_1"] = bilby.core.prior.Uniform(
    name='mass_1', minimum=5, maximum=60, unit='$M_{\\odot}$')
priors["mass_2"] = bilby.core.prior.Uniform(
    name='mass_2', minimum=5, maximum=60, unit='$M_{\\odot}$')
priors["eccentricity"] = bilby.core.prior.LogUniform(
    name='eccentricity', latex_label='$e$', minimum=1e-4, maximum=0.4)
priors["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(
    name='luminosity_distance', minimum=1e2, maximum=2e3)
priors["dec"] = bilby.core.prior.Cosine(name='dec')
priors["ra"] = bilby.core.prior.Uniform(
    name='ra', minimum=0, maximum=2 * np.pi)
priors["chi_1"] = bilby.core.prior.Uniform(
    name='chi_1', minimum=-1, maximum=0.6)
priors["chi_2"] = bilby.core.prior.Uniform(
    name='chi_2', minimum=-1, maximum=0.6)
priors["theta_jn"] = bilby.core.prior.Sine(name='theta_jn')
priors["psi"] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi)
priors["phase"] = bilby.core.prior.Uniform(
    name='phase', minimum=0, maximum=2 * np.pi)
priors["geocent_time"] = bilby.core.prior.Uniform(
    1180002600.9, 1180002601.1, name='geocent_time', unit='s')

# Initialising the likelihood function.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator)

# Now we run sampler (PyMultiNest in our case).
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

# And finally we make some plots of the output posteriors.
result.plot_corner()

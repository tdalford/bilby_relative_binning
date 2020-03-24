#!/usr/bin/env python
"""
A script to demonstrate how to use your own source model

Extra requirements
==================
- None!

Typical run time: ~ 1 minute
"""
import bilby
import numpy as np

# First set up logging and some output directories and labels
outdir = "outdir"
label = "create_your_own_source_model"
sampling_frequency = 256
duration = 1


# Here we define out source model - this is the sine-Gaussian model in the
# frequency domain.
def sine_gaussian(f, amplitude, peak_frequency, decay_time, phi0):
    arg = -((np.pi * decay_time * (f - peak_frequency)) ** 2) + 1j * phi0
    plus = np.sqrt(np.pi) * amplitude * decay_time * np.exp(arg) / 2.0
    cross = plus * np.exp(1j * np.pi / 2)
    return {"plus": plus, "cross": cross}


# We now define some parameters that we will inject
# Note that we define values for time, sky position, and polarisation
# which are not parameters of the model, but are essential for determining
# the detector response
injection_parameters = dict(
    amplitude=1e-23,
    peak_frequency=100,
    decay_time=1,
    phase=0,
    geocent_time=0,
    ra=0,
    dec=0,
    psi=0,
)

# Now we pass our source function to the WaveformGenerator
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=sine_gaussian,
)

# Set up interferometers.
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 3,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Here we define the priors for the search. We use the injection parameters
# except for the amplitude and peak_frequency.
prior = injection_parameters.copy()
prior["A"] = bilby.core.prior.LogUniform(
    minimum=1e-25, maximum=1e-21, latex_label="$A$"
)
prior["f0"] = bilby.core.prior.Uniform(minimum=90, maximum=110, latex_label="$f_0$")

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

result = bilby.core.sampler.run_sampler(
    likelihood,
    prior,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    nlive=100,
    sample="unif",
    injection_parameters=injection_parameters,
)
result.plot_corner()

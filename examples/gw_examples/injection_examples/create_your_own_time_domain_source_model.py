#!/usr/bin/env python
"""
A script to show how to create your own time domain source model.
A simple damped Gaussian signal is defined in the time domain, injected into
noise in two interferometers (LIGO Livingston and Hanford at design
sensitivity), and then recovered.

Extra requirements
==================
- None!

Typical run time: ~ 5 minutes
"""
import numpy as np
import bilby


def time_domain_damped_sinusoid(
    time, amplitude, damping_time, frequency, phase, start_time
):
    """
    This example only creates a linearly polarised signal with only plus
    polarisation.
    """
    plus = np.zeros(len(time))
    tidx = time >= start_time
    plus[tidx] = (
        amplitude
        * np.exp(-(time[tidx] - start_time) / damping_time)
        * np.sin(2 * np.pi * frequency * (time[tidx] - start_time) + phase)
    )
    return {"plus": plus}


# define parameters to inject.
injection_parameters = dict(
    amplitude=5e-22,
    damping_time=0.1,
    frequency=50,
    phase=0,
    ra=0,
    dec=0,
    psi=0,
    start_time=0.0,
    geocent_time=0.0,
)

duration = 1.0
sampling_frequency = 256
outdir = "outdir"
label = "time_domain_source_model"

# call the waveform_generator to create our waveform model.
waveform = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=time_domain_damped_sinusoid,
    start_time=injection_parameters["geocent_time"] - 0.5,
)

# inject the signal into three interferometers
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 0.5,
)
ifos.inject_signal(waveform_generator=waveform, parameters=injection_parameters)

#  create the priors
prior = injection_parameters.copy()
prior["amplitude"] = bilby.core.prior.LogUniform(
    minimum=1e-23, maximum=1e-21, latex_label="$h_0$"
)
prior["damping_time"] = bilby.core.prior.Uniform(
    minimum=0.01, maximum=0.2, latex_label="$\\tau$", unit="$s$"
)
prior["frequency"] = bilby.core.prior.Uniform(
    minimum=48, maximum=52, latex_label="$f$", unit="Hz"
)
prior["phase"] = bilby.core.prior.Uniform(
    minimum=-np.pi / 2, maximum=np.pi / 2, latex_label="$\\phi$"
)

# define likelihood
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform)

# launch sampler
result = bilby.core.sampler.run_sampler(
    likelihood,
    prior,
    sampler="dynesty",
    npoints=500,
    walks=10,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

result.plot_corner()

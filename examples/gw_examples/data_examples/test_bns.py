from __future__ import division, print_function
import os
import bilby
import numpy as np
import pickle

os.chdir('/Users/tommyalford/LIGO_research/bilby_relative_binning/bilby')

outdir = 'test_outdir'
label = 'GW170817'
time_of_event = bilby.gw.utils.get_event_time(label)
bilby.core.utils.setup_logger(outdir=outdir, label=label)
# GET DATA FROM INTERFEROMETER
# include 'V1' for appropriate O2 events
interferometer_names = ['H1']  # , 'L1', 'V1']
duration = 32
roll_off = 0.2  # how smooth is the transition from no signal
# to max signal in a Tukey Window.
psd_offset = -512  # PSD is estimated using data from
# `center_time+psd_offset` to `center_time+psd_offset + psd_duration`
# This determines the time window used to fetch open data.
psd_duration = 1024
coherence_test = False  # coherence between detectors
filter_freq = None  # low pass filter frequency to cut signal content above
# Nyquist frequency. The condition is 2 * filter_freq >= sampling_frequency


# All keyword arguments are passed to
# `gwpy.timeseries.TimeSeries.fetch_open_data()'
kwargs = {}
# Data are stored by LOSC at 4096 Hz, however
# there may be event-related data releases with a 16384 Hz rate.
kwargs['sample_rate'] = 4096
# For O2 events a "tag" is required to download the data.
# CLN = clean data; C02 = raw data
kwargs['tag'] = 'C02'
# interferometers = bilby.gw.detector.get_event_data(
#     label,
#     interferometer_names=interferometer_names,
#     duration=duration,
#     roll_off=roll_off,
#     psd_offset=psd_offset,
#     psd_duration=psd_duration,
#     cache=True,
#     filter_freq=filter_freq,
#     **kwargs)

interferometers = pickle.load(open('interferometer_data.pkl', 'rb'))
# CHOOSE PRIOR FILE
prior = bilby.gw.prior.BNSPriorDict(filename='GW170817.prior')
deltaT = 0.1
prior['geocent_time'] = bilby.core.prior.Uniform(
    minimum=time_of_event - deltaT / 2,
    maximum=time_of_event + deltaT / 2,
    name='geocent_time',
    latex_label='$t_c$',
    unit='$s$')


# GENERATE WAVEFORM
# OVERVIEW OF APPROXIMANTS:
# https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Waveforms/Overview
duration = None  # duration and sampling frequency will be overwritten
# to match the ones in interferometers.
sampling_frequency = kwargs['sample_rate']
start_time = 0  # set the starting time of the time array
waveform_arguments = {'reference_frequency': 20}

source_model = bilby.gw.source.lal_binary_neutron_star
convert_bns = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=source_model,
    parameter_conversion=convert_bns,
    waveform_arguments=waveform_arguments,)

init_params = {
    'chirp_mass': 1.1977, 'symmetric_mass_ratio': .244,
    'a_1': 0, 'a_2': 0, 'tilt_1': 0, 'tilt_2': 0,
    'phi_12': 0, 'phi_jl': 0, 'luminosity_distance': 50,
    'dec': 0, 'ra': 0, 'theta_jn': 0, 'psi': 0, 'phase': 0,
    # 'lambda_1': 0, 'lambda_2': 0,
    'geocent_time': 11870882.3848135}

parameter_bounds = {
    'chirp_mass': [1.18, 1.2], 'symmetric_mass_ratio': [.2, .2499],
    'a_1': [0, .05], 'a_2': [0, .05], 'tilt_1': [0, 2 * np.pi],
    'tilt_2': [0, 2 * np.pi], 'phi_12': [0, 2 * np.pi],
    'phi_jl': [0, 2 * np.pi], 'luminosity_distance': [10, 100],
    'dec': [0, 2 * np.pi], 'ra': [0, 2 * np.pi],
    'theta_jn': [0, 2 * np.pi], 'psi': [0, np.pi],
    'phase': [0, 2 * np.pi],
    # 'lambda_1': [0, 5000], 'lambda_2': [0, 5000],
    'geocent_time': [1187008881.3848135, 1187008883.3848135]}

# CHOOSE LIKELIHOOD FUNCTION
# Time marginalisation uses FFT.
# Distance marginalisation uses a look up table calculated at run time.
# Phase marginalisation is done analytically using a Bessel function.
likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    interferometers,
    waveform_generator,
    initial_parameters=init_params,
    parameter_bounds=parameter_bounds)

# RUN SAMPLER
# Implemented Samplers:
# LIST OF AVAILABLE SAMPLERS: Run -> bilby.sampler.implemented_samplers
# conversion function = bilby.gw.conversion.generate_all_bns_parameters
npoints = 512
sampler = 'dynesty'
result = bilby.run_sampler(
    likelihood,
    prior,
    outdir=outdir,
    label=label,
    sampler=sampler,
    npoints=npoints,
    use_ratio=False,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters)

# result.plot_corner()

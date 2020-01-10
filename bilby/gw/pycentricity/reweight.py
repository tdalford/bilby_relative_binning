"""

A file for reweighting non-SEOBNRe results using an SEOBNRe waveform for the pycentricity package.

"""
import matplotlib.pyplot as plt

import bilby as bb
import numpy as np
import pycentricity.overlap as ovlp
import pycentricity.waveform as wf
import numpy.random as random

from scipy.misc import logsumexp
from scipy.interpolate import interp1d
from scipy.special import i0e

from collections import namedtuple


_CalculatedSNRs = namedtuple('CalculatedSNRs',
                                 ['d_inner_h',
                                  'optimal_snr_squared',
                                  'complex_matched_filter_snr',
                                  'd_inner_h_squared_tc_array'])

def phase_marginalized_likelihood(d_inner_h, h_inner_h):
    # IRS TODO write docstring
    _bessel_function_interped = interp1d(
            np.logspace(-5, 10, int(1e6)), np.logspace(-5, 10, int(1e6)) +
            np.log([i0e(snr) for snr in np.logspace(-5, 10, int(1e6))]),
            bounds_error=False, fill_value=(0, np.nan))
    d_inner_h = _bessel_function_interped(abs(d_inner_h))
    return d_inner_h - h_inner_h / 2


def time_marginalized_likelihood(d_inner_h_tc_array, h_inner_h, priors, sampling_frequency, interferometers):
    # IRS TODO write docstring
    log_l_tc_array = phase_marginalized_likelihood(
                d_inner_h=d_inner_h_tc_array,
                h_inner_h=h_inner_h)
    #print('phase marginalised likelihood: {}'.format(log_l_tc_array))
    delta_tc = 2 / sampling_frequency
    times = interferometers.start_time + np.linspace(
                0, interferometers.duration,
                int(interferometers.duration / 2 *
                    sampling_frequency + 1))[1:]
    time_prior_array = priors['geocent_time'].prob(times) * delta_tc
    return logsumexp(log_l_tc_array, b=time_prior_array)


def calculate_snrs(waveform_polarizations, interferometer, parameters, duration):
    # IRS TODO add docstring 
    signal = interferometer.get_detector_response(waveform_polarizations, parameters)
    d_inner_h = interferometer.inner_product(signal=signal)
    optimal_snr_squared = interferometer.optimal_snr_squared(signal=signal)
    complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

    d_inner_h_squared_tc_array =\
                4 / duration * np.fft.fft(
                    signal[0:-1] *
                    interferometer.frequency_domain_strain.conjugate()[0:-1] /
                    interferometer.power_spectral_density_array[0:-1])
    return _CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_squared_tc_array=d_inner_h_squared_tc_array)

   

def phase_time_marginalized_log_likelihood_ratio(waveform_polarizations, interferometers, 
                                                 parameters, duration, priors, sampling_frequency):
    # IRS TODO add docstring 
    d_inner_h = 0.
    optimal_snr_squared = 0.
    complex_matched_filter_snr = 0.
    d_inner_h_tc_array = np.zeros(interferometers.frequency_array[0:-1].shape, dtype=np.complex128)

    for interferometer in interferometers:
        per_detector_snr = calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
                duration=duration)
        #print('SNR for {}: {}'.format(interferometer.name, per_detector_snr))

        d_inner_h += per_detector_snr.d_inner_h
        optimal_snr_squared += np.real(per_detector_snr.optimal_snr_squared)
        complex_matched_filter_snr += per_detector_snr.complex_matched_filter_snr

        d_inner_h_tc_array += per_detector_snr.d_inner_h_squared_tc_array
    log_l = time_marginalized_likelihood(
                d_inner_h_tc_array=d_inner_h_tc_array,
                h_inner_h=optimal_snr_squared,
                priors=priors,
                sampling_frequency=sampling_frequency, 
                interferometers=interferometers)
    #print('time and phase marginalised likelihood: {}'.format(log_l))
    #print('real part: {}'.format(float(log_l.real)))
    return float(log_l.real)


def log_likelihood_interferometer(
    waveform_polarizations, interferometer, parameters, duration
):
    """
    Return the log likelihood at a single interferometer.
    :param waveform_polarizations: dict
        frequency-domain waveform polarisations
    :param interferometer: Interferometer
        Interferometer object
    :param parameters: dict
        waveform parameters
    :param duration: int
        time duration of the signal
    :return:
        log_l: float
            log likelihood evaluation
    """
    signal_ifo = interferometer.get_detector_response(
        waveform_polarizations, parameters
    )
    log_l = (
        -2.0
        / duration
        * np.vdot(
            interferometer.frequency_domain_strain - signal_ifo,
            (interferometer.frequency_domain_strain - signal_ifo)
            / interferometer.power_spectral_density_array,
        )
    )
    return log_l.real


def noise_log_likelihood(interferometers, duration):
    """
    Return the likelihood that the data is caused by noise
    :param interferometers: InterferometerList
        list of interferometers involved in the detection
    :param duration: int
        time duration of the signal
    :return:
        log_l: float
        log likelihood evaluation
    """
    log_l = 0
    for interferometer in interferometers:
        log_l -= (
            2.0
            / duration
            * np.sum(
                abs(interferometer.frequency_domain_strain) ** 2
                / interferometer.power_spectral_density_array
            )
        )
    return log_l.real


def log_likelihood(waveform_polarizations, interferometers, parameters, duration):
    """
    Return the log likelihood of all interferometers involved in the detection
    :param waveform_polarizations: dict
        frequency-domain waveform polarisations
    :param interferometers: InterferometerList
        list of interferometers involved in the detection
    :param parameters: dict
        waveform parameters
    :param duration: int
        time duration of the signal
    :return:
        log_l: float
        log likelihood evaluation
    """
    log_l = 0
    for interferometer in interferometers:
        log_l += log_likelihood_interferometer(
            waveform_polarizations, interferometer, parameters, duration
        )
    return log_l.real


def log_likelihood_ratio(waveform_polarizations, interferometers, parameters, duration):
    """
    Return the ratio of the likelihood of the signal being real to the likelihood
    of the signal being noise
    :param waveform_polarizations: dict
        frequency-domain waveform polarisations
    :param interferometers: InterferometerList
        list of interferometers involved in the relationship
    :param parameters: dict
        waveform variables
    :param duration: int
        time duration of the signal
    :return:
        log_likelihood_ratio: float
            log likelihood evaluation ratio
    """
    return log_likelihood(
        waveform_polarizations, interferometers, parameters, duration
    ) - noise_log_likelihood(interferometers, duration)


def calculate_log_weight(eccentric_log_likelihood_array, circular_log_likelihood):
    """
    Return the log weight for the sample with the passed-in eccentric likelihood array.
    :param eccentric_log_likelihood_array: array/list
        list of eccentric log-likelihoods
    :param circular_log_likelihood: float
        value of the circular log-likelihood
    :return:
        average_log_weight: float
            the log weight for the sample
    """
    individual_log_weight = [ll - circular_log_likelihood for ll in eccentric_log_likelihood_array]
    average_weight = np.mean(np.exp(individual_log_weight))
    average_log_weight = np.log(average_weight)
    return average_log_weight


def pick_weighted_random_eccentricity(cumulative_density_grid, eccentricity_grid):
    """
    Return a random eccentricity, weighted by a cumulative density function.
    :param cumulative_density_grid: array
        1D grid of cumulative densities
    :param eccentricity_grid: array
        1D grid of eccentricities
    :return:
        random_eccentricity: float
            the weighted random eccentricity chosen
    """
    # First select a bin, weighted at random
    start = cumulative_density_grid[0]
    end = cumulative_density_grid[-1]
    random_value = random.random_sample() * (end - start) + start
    for i, cd in enumerate(cumulative_density_grid):
        if cd >= random_value:
            # Now select an eccentricity at random from within the selected bin
            lower_bound = 0
            upper_bound = eccentricity_grid[i]
            if i > 0:
                lower_bound = eccentricity_grid[i - 1]
            random_eccentricity = (
                random.random_sample() * (upper_bound - lower_bound) + lower_bound
            )
            return random_eccentricity


def cumulative_density_function(log_likelihood_grid):
    """
    Return a cumulative density function over a grid of eccentricities.
    :param log_likelihood_grid: array
        1D grid of log likelihoods
    :return:
    cumulative_density: array
        1D grid of cumulative densities
    """
    # Deal with extremely high values of log likelihood
    maximum_log_likelihood = np.max(log_likelihood_grid)
    # Ratio of likelihood to maximum log likelihood
    log_likelihood_grid = log_likelihood_grid - maximum_log_likelihood
    likelihood_grid = np.exp(log_likelihood_grid)
    cumulative_density =  np.cumsum(likelihood_grid)
    cumulative_density_normalised = cumulative_density / cumulative_density[-1]
    return cumulative_density_normalised


def deal_with_no_waveform_generation(label, eccentricity):
    # TODO IRS docstring
    print('No waveform generated; disregard sample {}'.format(label))
    intermediate_outfile.write("{}\t\t{}\t\t{}\n".format(eccentricity, None, None))
    disregard = True
    return disregard


def prepare_updated_seobnre_waveform(eccentricity, parameters, sampling_frequency, seobnre_waveform_minimum_frequency):
    # TODO IRS docstring
   #print('seobnre waveform minimum frequency: {} Hz'.format(seobnre_waveform_minimum_frequency))
   parameters.update({"eccentricity": eccentricity})
   # Need to have a set minimum frequency, since this is also the reference frequency
   seobnre_waveform_time_domain = wf.seobnre_bbh_with_spin_and_eccentricity(
            parameters=parameters,
            sampling_frequency=sampling_frequency,
            minimum_frequency=seobnre_waveform_minimum_frequency,
   )
   return seobnre_waveform_time_domain


def obtain_output_parameters(log_likelihood_grid, eccentricity_grid, original_log_likelihood):
    # TODO IRS docstring
    cumulative_density_grid = cumulative_density_function(log_likelihood_grid)
    # We want to pick a weighted random point from within the CDF
    new_e = pick_weighted_random_eccentricity(cumulative_density_grid, eccentricity_grid)
    # Also return average log-likelihood
    average_log_likelihood = np.log(np.sum(np.exp(log_likelihood_grid)) / len(log_likelihood_grid))
    
    # Weight calculated using average likelihood
    log_weight = calculate_log_weight(log_likelihood_grid, original_log_likelihood)
    #print('new e: {}'.format(new_e))
    return new_e, average_log_likelihood, log_weight 


def set_up_intermediate_output_file(label, parameters):
   # IRS TODO docstring
    intermediate_outfile = open("{}_eccentricity_result.txt".format(label), "w")
    intermediate_outfile.write("sample parameters:\n")
    for key in parameters.keys():
        intermediate_outfile.write("{}:\t{}\n".format(key, parameters[key]))
    intermediate_outfile.write("\n-------------------------\n")
    intermediate_outfile.write("e\t\tlog_L\t\tmaximised_overlap\n")
    return intermediate_outfile


def new_weight(parameters, initial_log_L, interferometers, duration, minimum_frequency, 
               sampling_frequency, maximum_frequency, priors, minimum_log_eccentricity, 
               maximum_log_eccentricity, signal_length, number_of_eccentricity_bins, comparison_waveform, label,
               seobnre_waveform_minimum_frequency
    ):
    # IRS TODO add docstring
    eccentricity_grid = np.logspace(
        minimum_log_eccentricity, maximum_log_eccentricity, number_of_eccentricity_bins
    ) 
   
    log_likelihood_grid = []
    disregard = False
    intermediate_outfile = set_up_intermediate_output_file(label, parameters)
    # Need to update parameters with phase and time appropriate for marginalization
    parameters.update({'phase': 0, 'geocent_time': interferometers.start_time})
    recalculated_circular_log_L = phase_time_marginalized_log_likelihood_ratio(
                               waveform_polarizations=comparison_waveform, interferometers=interferometers,
                              parameters=parameters, duration=duration, priors=priors,
                              sampling_frequency=sampling_frequency)

    recalculated_log_l_differs_from_original_significantly_warning(recalculated_circular_log_L, initial_log_L)

    for e in eccentricity_grid:
        seobnre_waveform_time_domain = prepare_updated_seobnre_waveform(e, parameters, sampling_frequency, seobnre_waveform_minimum_frequency)
        if seobnre_waveform_time_domain['plus'] is None:
            disregard = deal_with_no_waveform_generation(label, e)
        else:
            seobnre_waveform_time_domain = ovlp.apply_tukey_window(seobnre_waveform_time_domain, alpha=0.05)
            # Do the signal processing
            seobnre_waveform_time_domain = ovlp.process_signal(seobnre_waveform_time_domain, comparison_length=signal_length)
            # Wrap at final index for consistency with recovery waveform
            seobnre_waveform_time_domain, initial_shift = ovlp.wrap_at_maximum(
               seobnre_waveform_time_domain, (signal_length-1)
            )
            # Do the Fourier transform
            seobnre_waveform_frequency_domain = ovlp.fourier_transform(seobnre_waveform_time_domain, sampling_frequency)
            # Calculate eccentric log likelihood
            eccentric_log_L = phase_time_marginalized_log_likelihood_ratio(
                               waveform_polarizations=seobnre_waveform_frequency_domain, interferometers=interferometers, 
                              parameters=parameters, duration=duration, priors=priors, 
                              sampling_frequency=sampling_frequency)
            # Store log likelihood 
            log_likelihood_grid.append(eccentric_log_L)
            # Write out - no overlap
            intermediate_outfile.write(
                "{}\t\t{}\t\tNA\n".format(e, eccentric_log_L)
            )
    intermediate_outfile.close()
    if not disregard:
        return obtain_output_parameters(log_likelihood_grid, eccentricity_grid, recalculated_circular_log_L)
    else:
        return None, None, None 
   

def recalculated_log_l_differs_from_original_significantly_warning(recalculated_log_likelihood, log_L):
    # Print a warning if this is much different to the likelihood stored in the results
    if abs(recalculated_log_likelihood - log_L) / log_L > 0.1:
        percentage = abs(recalculated_log_likelihood - log_L) / log_L * 100
        print(
            "WARNING :: recalculated log likelihood differs from original by {}%".format(
                percentage
            )
        )
        print('original log L: {}'.format(log_L))
        print('recalculated log L: {}'.format(recalculated_log_likelihood))


def new_weight_phase_time_maximised(
    log_L,
    parameters,
    comparison_waveform_frequency_domain,
    interferometers,
    duration,
    sampling_frequency,
    minimum_frequency,
    maximum_frequency,
    label,
    minimum_log_eccentricity=-4,
    maximum_log_eccentricity=np.log10(0.2),
    number_of_eccentricity_bins=40,
    reference_frequency=10
):
    """
    Compute the new weight for a point, weighted by the eccentricity-marginalised likelihood.
    :param log_L: float
        the original likelihood of the point
    :param parameters: dict
        the parameters that define the sample
    :param comparison_waveform_frequency_domain: dict
        frequency-domain waveform polarisations of the waveform used for the original analysis
    :param interferometers: InterferometerList
        list of interferometers used in the detection
    :param duration: int
        time duration of the signal
    :param sampling_frequency: int
        the frequency with which to 'sample' the waveform
    :param maximum_frequency: int
        the maximum frequency at which the data is analysed
    :param label: str
        identifier for results
    :param minimum_log_eccentricity: float
        minimum eccentricity to look for
    :param maximum_log_eccentricity: float
        maximum log eccentricity to look for
    :param number_of_eccentricity_bins: int
        granularity of search in eccentricity space
    :param reference_frequency: float
        frequency at which to measure eccentricity
    :return:
        e: float
            the new eccentricity sample
        average_log_likelihood: float
            the eccentricity-marginalised new likelihood
        log_weight: float
            the log weight of the sample
    """
    # First calculate a grid of likelihoods.
    grid_size = number_of_eccentricity_bins
    eccentricity_grid = np.logspace(
        minimum_log_eccentricity, maximum_log_eccentricity, grid_size
    )
    # Recalculate the log likelihood of the original sample
    recalculated_log_likelihood = log_likelihood_ratio(
        comparison_waveform_frequency_domain, interferometers, parameters, duration
    )
    # Print a warning if this is much different to the likelihood stored in the results
    recalculated_log_l_differs_from_original_significantly_warning(recalculated_log_likelihood, log_L)

    log_likelihood_grid = []

    intermediate_outfile = set_up_intermediate_output_file(label, parameters)    

    # Prepare for the possibility that we have to disregard this sample
    disregard = False

    for e in eccentricity_grid:
        seobnre_waveform_time_domain = prepare_updated_seobnre_waveform(e, parameters, sampling_frequency, reference_frequency) 
        if seobnre_waveform_time_domain['plus'] is None:
            disregard = deal_with_no_waveform_generation(label, e) 
        else:
            seobnre_wf_td, seobnre_wf_fd, max_overlap, index_shift, phase_shift = ovlp.maximise_overlap(
                seobnre_waveform_time_domain,
                comparison_waveform_frequency_domain,
                sampling_frequency,
                interferometers[0].frequency_array,
                interferometers[0].power_spectral_density,
                minimum_frequency=interferometers[0].minimum_frequency,
                maximum_frequency=interferometers[0].maximum_frequency
            )
            eccentric_log_L = log_likelihood_ratio(
                seobnre_wf_fd, interferometers, parameters, duration
            )
            log_likelihood_grid.append(eccentric_log_L)
            intermediate_outfile.write(
                "{}\t\t{}\t\t{}\n".format(e, eccentric_log_L, max_overlap)
            )
    intermediate_outfile.close()
 
    if not disregard:
        return obtain_output_parameters(log_likelihood_grid, eccentricity_grid, recalculated_log_likelihood)
    else:
        return None, None, None


def output_reweighting_progress(i, number_of_samples):
    # IRS TODO docstring
    print(
            "new weight calculation {}% complete".format(
                np.round((i + 1) / number_of_samples * 100, 2)
         )
    )


def write_results_to_output(i, eccentricity, new_log_L, log_weight, output, outfile):
    # IRS TODO docstring
    outfile.write("{}\t\t{}\t\t{}\t\t{}\n".format(i, eccentricity, new_log_L, log_weight))
    output["eccentricity"].append(eccentricity)
    output["new_log_L"].append(new_log_L)
    output["log_weight"].append(log_weight)


def deal_with_high_spins(i, chi_1, chi_2, output, outfile):
    # IRS TODO docstring  
   print('omitting sample; chi_1 = {}, chi_2 = {}'.format(chi_1, chi_2))
   output["eccentricity"].append(None)
   output["new_log_L"].append(None)
   output["log_weight"].append(None)
   outfile.write("{}\t\t{}\t\t{}\t\t{}\n".format(i, None, None, None))


def set_up_outfile(output_folder, label):
    # IRS TODO docstring
    outfile = open("{}/{}_master_output_store.txt".format(output_folder, label), "w")
    outfile.write("i\t\te\t\tnew_log_L\t\tlog_w\n")
    return outfile


def get_converted_samples(samples):
    # IRS TODO add docstring
    converted_samples, added_keys = bb.gw.conversion.convert_to_lal_binary_black_hole_parameters(
        samples
    )
    converted_samples = {
        key: converted_samples[key].values.tolist()
        for key in converted_samples.keys()
        if type(converted_samples[key]) is not float
    }
    return converted_samples


def get_parameter_list(converted_samples, minimum_log_eccentricity):
    # IRS TODO add docstring
    parameter_list = [
        dict(
            mass_1=converted_samples["mass_1"][i][0],
            mass_2=converted_samples["mass_2"][i][0],
            luminosity_distance=converted_samples["luminosity_distance"][i][0],
            geocent_time=converted_samples["geocent_time"][i][0],
            ra=converted_samples["ra"][i][0],
            dec=converted_samples["dec"][i][0],
            theta_jn=converted_samples["theta_jn"][i][0],
            psi=converted_samples["psi"][i][0],
            phase=converted_samples["phase"][i][0],
            chi_1=converted_samples["chi_1"][i][0],
            chi_2=converted_samples["chi_2"][i][0],
            eccentricity=np.power(10.0, minimum_log_eccentricity),
        )
	for i in range(len(converted_samples['chi_1']))
    ]
    return parameter_list


def reweight_by_eccentricity(samples, log_likelihood, minimum_frequency, reference_frequency, maximum_frequency, 
                             sampling_frequency, duration, interferometers, priors, minimum_log_eccentricity, 
                             maximum_log_eccentricity, number_of_eccentricity_bins, comparison_waveform_generator,
                             label, output_folder, seobnre_waveform_minimum_frequency=10):
    # IRS TODO add docstring
    number_of_samples = len(log_likelihood)
    converted_samples = get_converted_samples(samples)    
    parameter_list = get_parameter_list(converted_samples, minimum_log_eccentricity)
    signal_length = int(duration * sampling_frequency)

    
    # Generate the IMRPhenomD waveform for each sample
    comparison_waveform_strain_list = [
        comparison_waveform_generator.frequency_domain_strain(parameters)
        for parameters in parameter_list
    ]

    output = {key: [] for key in ["eccentricity", "new_log_L", "log_weight"]}
 
    # Write the output file along the way
    outfile = set_up_outfile(output_folder, label)
    for i, log_L in enumerate(log_likelihood):
        # debugging
        #print(parameter_list[i])
        # If the spins are too large, the sample may fail to generate eccentric waveforms,
        # so we impose a moderate-spin prior here
        if any([parameter_list[i]['chi_1'] > 0.6, parameter_list[i]['chi_2'] > 0.6]):
            deal_with_high_spins(i, parameter_list[i]['chi_1'], parameter_list[i]['chi_2'], output, outfile)
        else:
            eccentricity, new_log_L, log_weight = new_weight(
               parameters=parameter_list[i] , initial_log_L=log_L, interferometers=interferometers, 
               duration=duration, minimum_frequency=minimum_frequency, sampling_frequency=sampling_frequency, 
               maximum_frequency=maximum_frequency, priors=priors, minimum_log_eccentricity=minimum_log_eccentricity,
               maximum_log_eccentricity=maximum_log_eccentricity, signal_length=signal_length,
               number_of_eccentricity_bins=number_of_eccentricity_bins, comparison_waveform=comparison_waveform_strain_list[i], label="{}/{}_{}".format(output_folder, label, i),
               seobnre_waveform_minimum_frequency=seobnre_waveform_minimum_frequency
            )
            write_results_to_output(i, eccentricity, new_log_L, log_weight, output, outfile)
        output_reweighting_progress(i, number_of_samples)
    outfile.close()
    return output       
    


def reweight_by_eccentricity_phase_time_maximised(
    samples,
    log_likelihood,
    sampling_frequency,
    comparison_waveform_generator,
    interferometers,
    duration,
    output_folder=".",
    minimum_frequency=20,
    maximum_frequency=1024,
    label="",
    minimum_log_eccentricity=-4,
    maximum_log_eccentricity=np.log10(0.2),
    number_of_eccentricity_bins=20,
    reference_frequency=20
):
    """
    Function to return a dictionary containing the eccentricity-marginalised log likelihood,
    the weight of the sample and the new eccentricity.
    :param samples: dict
        dictionary of all posterior samples from the original analysis
    :param log_likelihood: list
        list of log-likelihoods from the original analysis
    :param sampling_frequency: int
        the frequency at which the data is sampled
    :param minimum_frequency: int
        the minimum frequency at which the data is sampled
    :param comparison_waveform_generator: WaveformGenerator
        the waveform generator for the waveform used in the original analysis
    :param interferometers: InterferometerList
        list of interferometers used in the detection
    :param duration: int
        time duration of the analysis
    :param output_folder: str
        location to send the output
    :param maximum_frequency: int
        the maximum frequency of the analysis
    :param label: str
        identifier for results
    :param minimum_log_eccentricity: float
        minimum eccentricity to look for
    :param maximum_log_eccentricity: float
        maximum log eccentricity to look for
    :param number_of_eccentricity_bins: int
        granularity of search in eccentricity space
    :param reference_frequency: float
        frequency at which to measure eccentricity
    :return:
        output: dict
            dictionary of output from the reweighting procedure
    """
    number_of_samples = len(log_likelihood)
    converted_samples = get_converted_samples(samples)
    parameter_list = get_parameter_list(converted_samples, minimum_log_eccentricity)

    # Generate the IMRPhenomD waveform for each sample
    comparison_waveform_strain_list = [
        comparison_waveform_generator.frequency_domain_strain(parameters)
        for parameters in parameter_list
    ]

    output = {key: [] for key in ["eccentricity", "new_log_L", "log_weight"]}

    # Write the output file along the way
    outfile = set_up_outfile(output_folder, label)
    for i, log_L in enumerate(log_likelihood):
        # If the spins are too large, the sample may fail to generate eccentric waveforms,
        # so we impose a moderate-spin prior here
        if any([parameter_list[i]['chi_1'] > 0.6, parameter_list[i]['chi_2'] > 0.6]):
             deal_with_high_spins(i, parameter_list[i]['chi_1'], parameter_list[i]['chi_2'], output, outfile)
        else:
            eccentricity, new_log_L, log_weight = new_weight_phase_time_maximised(
                log_L,
                parameter_list[i],
                comparison_waveform_strain_list[i],
                interferometers,
                duration,
                sampling_frequency,
                minimum_frequency,
                maximum_frequency,
                "{}/{}_{}".format(output_folder, label, i),
                minimum_log_eccentricity=minimum_log_eccentricity,
                maximum_log_eccentricity=maximum_log_eccentricity,
                number_of_eccentricity_bins=number_of_eccentricity_bins,
                reference_frequency=reference_frequency
            )
            write_results_to_output(i, eccentricity, new_log_L, log_weight, output, outfile)
        output_reweighting_progress(i, number_of_samples)
    outfile.close()
    return output

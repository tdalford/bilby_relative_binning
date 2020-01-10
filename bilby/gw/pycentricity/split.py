"""

A file for splitting a results file into the required subset for the pycentricity package.

"""

import pycentricity.GWTC1_utils as utils
import bilby as bb
import numpy as np
import json


bilby_to_lalinference = dict(
        a_1="a1",
        a_2="a2",
        chirp_mass="mc",
        chirp_mass_source="mc_source",
        total_mass="mtotal",
        luminosity_distance="dist",
        ra="ra",
        dec="dec",
        redshift="redshift",
        chi_eff="chi_eff",
        chi_1="a1z",
        chi_2="a2z",
        psi="psi",
        mass_ratio="q",
        geocent_time="time",
        theta_jn="theta_jn",
        phase="phase_maxl",
        symmetric_mass_ratio="eta",
        mass_1="m1",
        mass_1_source="m1_source",
        mass_2="m2",
        mass_2_source="m2_source",
        log_likelihood="logl",
        noise_log_likelihood="nulllogl"
)


def read_lalinference_result(result_file):
    init_lalinf = np.genfromtxt(result_file, names=True)
    print(init_lalinf.dtype.names)
    lalinf = {}
    for key in bilby_to_lalinference.keys():
        try:
            lalinf[key] = init_lalinf[bilby_to_lalinference[key]]
        except ValueError:
            print("could not match key {} with string {}".format(key, bilby_to_lalinference[key]))
            continue
    return lalinf


def result_subset(start_index, end_index, result):
    """
    Return a certain subset of a set of results.
    :param start_index: int
        the index at which to begin the subset
    :param end_index: int
        the index at which to end the subset
    :param result: Result
        the result object
    :return:
        results_subset_dictionary: dict
            dictionary containing information about the result subset
    """
    samples = result.posterior
    keys = utils.search_keys
    log_likelihood = result.posterior.log_likelihood
    subset_samples = {key: samples[key][start_index:end_index].tolist() for key in keys}
    subset_log_likelihood = log_likelihood[start_index:end_index].tolist()
    results_subset_dictionary = dict(
        samples=subset_samples, log_likelihoods=subset_log_likelihood
    )
    return results_subset_dictionary


def result_subset_from_lalinference(start_index, end_index, result):
    """
    Return a certain subset of a set of results.
    :param start_index: int
        the index at which to begin the subset
    :param end_index: int
        the index at which to end the subset
    :param result: Result
        the result object
    :return:
	results_subset_dictionary: dict
            dictionary containing information about the result subset
    """
    keys = list(utils.parameter_keys.keys())
    log_likelihood = result['log_likelihood'] - result['noise_log_likelihood']
    subset_samples = {key: result[key][start_index:end_index].tolist() for key in keys if key in bilby_to_lalinference}
    subset_log_likelihood = log_likelihood[start_index:end_index].tolist()
    results_subset_dictionary = dict(
        samples=subset_samples, log_likelihoods=subset_log_likelihood
    )
    return results_subset_dictionary


def write_subset_to_file(start_index, end_index, result, index, output_file_path, pipeline='bilby'):
    """
    Save a certain subset to a file.
    :param start_index: int
        the index at which to begin the subset
    :param end_index: int
        the index at which to end the subset
    :param result: Result
        the result object
    :param output_file_path: str
        the path to the location in which to create output files
    """
    if pipeline == 'bilby':
        result_subset_dictionary = result_subset(start_index, end_index, result)
    else:
        result_subset_dictionary = result_subset_from_lalinference(start_index, end_index, result)
    output_file_path += "/result_{}.json".format(index)
    with open(output_file_path, "w") as f:
        json.dump(result_subset_dictionary, f)


def split_results_into_subsets(number_per_file, result_file, pipeline='bilby'):
    """
    Split a set of results into numerous subsets for easier computation
    :param number_per_file: int
        number of results to store per file
    :param result_file: str
        file path of results file
    """
    # Work out where to store the output
    output_file_path_list = result_file.split("/")
    output_file_path = ""
    for string in output_file_path_list[0:-1]:
        output_file_path += string + "/"
    output_file_path += "subsets/"
    bb.core.utils.check_directory_exists_and_if_not_mkdir(output_file_path)
    # Get the result object
    if pipeline == 'bilby':
        result = bb.result.read_in_result(result_file)
        total_number_of_samples = len(result.posterior.log_likelihood)
    else:
        result = read_lalinference_result(result_file)
        total_number_of_samples = len(result['log_likelihood'])
    start_indices = np.arange(0, total_number_of_samples, number_per_file)
    for i, start_index in enumerate(start_indices):
        end_index = min(start_index + number_per_file, total_number_of_samples)
        write_subset_to_file(start_index, end_index, result, i, output_file_path, pipeline)
    print("results split into {} subsets".format(i))

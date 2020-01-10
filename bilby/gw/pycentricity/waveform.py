"""

A file for generating waveforms and related objects for the pycentricity package.

"""
from ctypes import *
from copy import deepcopy
import bilby as bb
import numpy as np
import os


def new_waveform_list(old_c_list):
    """
    Transform a c-type list to a python list.
    :param old_c_list: list of c_double
    :return:
        new_list: list
            python list
    """
    new_list = []
    if old_c_list is not None:
        for element in old_c_list:
            if element == 0.0:
                break
            new_list.append(element)
    return new_list


def generate_seobnre_waveform_from_c_code(parameters, deltaT, minimum_frequency):
    """
    Return the waveform polarisations simulated by the SEOBNRe c-code.
    :param parameters: dict
        dictionary of waveform parameters
    :param deltaT: float
        time step
    :param minimum_frequency: int
        minimum frequency to contain in the signal
    :return:
        hp, hc: c_double[]
            time-domain plus and cross waveform polarisations
    """
    seobnre = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/_seobnre.so')
    if parameters['chi_1'] > 0.6 or parameters['chi_2'] > 0.6:
        print('WARNING:: Dimensionless component spins larger than 0.6.')
        print('WARNING:: Not attempting to generate waveform. ')
        return None, None
    else:
        c_injection_parameters = deepcopy(parameters)
        for key in parameters:
            c_injection_parameters[key] = c_double(c_injection_parameters[key])
        c_deltaT = c_double(deltaT)
        c_minimum_frequency = c_double(minimum_frequency)
        hp = POINTER(c_double)()
        hc = POINTER(c_double)()
        seobnre.do_waveform_generation.argtypes = [
            POINTER(POINTER(c_double)),
            POINTER(POINTER(c_double)),
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
        ]
        seobnre.do_waveform_generation(
            pointer(hp),
            pointer(hc),
            c_injection_parameters['phase'],
            c_deltaT,
            c_injection_parameters['mass_1'],
            c_injection_parameters['mass_2'],
            c_injection_parameters['chi_1'],
            c_injection_parameters['chi_2'],
            c_minimum_frequency,
            c_injection_parameters['eccentricity'],
            c_injection_parameters['luminosity_distance'],
            c_injection_parameters['theta_jn']
        )
        return hp, hc


def seobnre_bbh_with_spin_and_eccentricity(
    parameters, sampling_frequency, minimum_frequency
):
    """
    Return the  waveform polarisations.
    :param parameters: dict
        dictionary of waveform parameters
    :param sampling_frequency: int
        frequency with which to 'sample' the waveform
    :param minimum_frequency: int
        minimum frequency to contain in the signal
    :return:
        seobnre: dict
            time-domain waveform polarisations
    """
    deltaT = 1 / sampling_frequency
    hp, hc = generate_seobnre_waveform_from_c_code(parameters, deltaT, minimum_frequency)
    seobnre_waveform = {
        "plus": new_waveform_list(hp),
        "cross": new_waveform_list(hc)
    }
    return seobnre_waveform


def get_comparison_waveform_generator(
    minimum_frequency, sampling_frequency, duration,
        maximum_frequency, reference_frequency, waveform_approximant="IMRPhenomD"
):
    """
    Provide the waveform generator object for the comparison waveform,
    which can be generically chosen.
    :param minimum_frequency: int
        minimum frequency to contain in the waveform
    :param sampling_frequency: int
        frequency with which to 'sample' the signal
    :param duration: int
        time duration of the signal
    :return:
        waveform_generator: WaveformGenerator
            the waveform generator object for the comparison waveform
    """
    waveform_arguments = dict(
        waveform_approximant=waveform_approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency
    )
    waveform_generator = bb.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bb.gw.source.lal_binary_black_hole,
        parameter_conversion=bb.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    return waveform_generator

import numpy as np
import pdb


class likelihood:
    def __init__(self, interferometers, source, parameters):
        self.interferometers = interferometers
        self.source = source
        self.parameters = parameters
        self.parameter_keys = ['f']

    def logl(self, theta):
        logL = 0
        self.parameters['f'] = theta[0]
        waveform_polarizations = self.source.frequency_domain_strain(self.parameters)
        for interferometer in self.interferometers:
            for mode in waveform_polarizations:
                det_response = interferometer.antenna_response(
                                self.parameters['ra'], self.parameters['dec'],
                                self.parameters['geocent_time'],
                                self.parameters['psi'], mode)

                waveform_polarizations[mode] *= det_response

            signal_IFO = np.sum(waveform_polarizations.values(), axis=0)

            #time_shift = interferometer.time_shift(source.params['geocent_time'])
            #signal *= np.exp(-1j*2*np.pi*time_shift)
            # This is just here as a reminder that a tc shift needs to be performed
            # on frequency-domain GWs

            logL -= 4. * self.source.sampling_frequency * np.vdot(
                interferometer.data - signal_IFO,
                (interferometer.data - signal_IFO) / interferometer.power_spectral_density_array)

        return logL.real


def logl_gravitational_wave(params, source, interferometers):
    """
    Calculate the log likelihood of a given gravitational-wave source given some strain data.

    :param sampling_frequency: data sampling frequency
    :param params: dictionary of parameter values describing the source
    :param source: Source object which contains a model
    :param interferometers: Interferometer classes with PSDs and data
    :return: log likelihood of the strain given the model and parameters
    """
    log_l = 0
    waveform_polarizations = source.frequency_domain_strain(params)
    for interferometer in interferometers:
        for mode in waveform_polarizations.keys():

            det_response = interferometer.antenna_response(params['ra'], params['dec'], params['geocent_time'],
                                                           params['psi'], mode)

            waveform_polarizations[mode] *= det_response

        signal_ifo = np.sum(waveform_polarizations.values(), axis=0)

        time_shift = interferometer.time_delay_from_geocenter(params['ra'], params['dec'], params['geocent_time'])
        signal_ifo *= np.exp(-1j*2*np.pi*time_shift)
        signal_ifo_whitened = signal_ifo / interferometer.amplitude_spectral_density_array

        log_l -= 4. * source.sampling_frequency * np.real(sum((interferometer.whitened_data - signal_ifo_whitened)**2))

    return log_l

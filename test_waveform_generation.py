from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.source import lal_eccentric_binary_black_hole_with_spins

waveform_arguments = dict(waveform_approximant='SEOBNRE', minimum_frequency=10, sampling_frequency=1024)
parameters = dict(mass_1=30, mass_2=28, eccentricity=0.1, chi_1=0.0, chi_2=0.0, theta_jn=0.0, luminosity_distance=420, phase=0.0)

waveform_gen = WaveformGenerator(
                 sampling_frequency=waveform_arguments['sampling_frequency'],
                 duration=4,
                 time_domain_source_model=lal_eccentric_binary_black_hole_with_spins,
                 waveform_arguments=waveform_arguments,
               )

time_domain_strain = waveform_gen.time_domain_strain(parameters)

import matplotlib.pyplot as plt

figure = plt.figure()
plt.plot(time_domain_strain['plus'], label='plus')
plt.plot(time_domain_strain['cross'], label='cross')
plt.savefig('/home/isobel.romero-shaw/public_html/seobnre_test_waveform.png', bbox_inches='tight')
plt.clf()


"""

A file containing plotting functions for the pycentricity package.

"""
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams

import corner
import pycentricity.GWTC1_utils as utils
import pycentricity.waveform as wf
import pycentricity.overlap as ovlp

import wquantiles as wq
import bilby as bb

fontparams = {'mathtext.fontset': 'stix',
             'font.family': 'serif',
             'font.serif': "Times New Roman",
             'mathtext.rm': "Times New Roman",
             'mathtext.it': "Times New Roman:italic",
             'mathtext.sf': 'Times New Roman',
             'mathtext.tt': 'Times New Roman'}
rcParams.update(fontparams)


def get_data(data_list):
    """
    Simply transpose and stack a list of data.
    :param data_list: list
        list of data to transpose and stack
    :return:
        transposed and stacked list
    """
    return np.transpose(np.vstack(data_list))


def update_log_eccentricity_bins_and_range_for_corner(
    minimum_log_eccentricity, maximum_log_eccentricity, number_of_bins, subset_keys, subset
):
    bins = []
    range = []
    for k in subset_keys[subset]:
        if 'log_eccentricity' in k:
            bins.append(np.linspace(minimum_log_eccentricity, maximum_log_eccentricity, number_of_bins))
            range.append([minimum_log_eccentricity, maximum_log_eccentricity])
        else:
            bins.append(number_of_bins)
            range.append(0.9999)
    return bins, range


def plot_corner_weights_against_parameter(log_weights, other_parameter_array, parameter_name, label):
   data = get_data([log_weights, other_parameter_array])
   corner.corner(data, labels=['$ln w$', utils.parameter_keys[parameter_name]])
   plt.savefig('{}_weight_{}_corner.png'.format(label, parameter_name), bbox_inches='tight')

def plot_corner_weights_eccentricities(log_weights, log_eccentricities, label):
   data = get_data([log_weights, log_eccentricities])
   corner.corner(data, labels=['$ln w$', '$log_{10}e$'])
   plt.savefig('{}_weight_eccentricity_corner.png'.format(label), bbox_inches='tight')
   
     
def plot_reweighted_posteriors(
        posterior, eccentricities, new_weights, label, original_weights=None, injection_values=None, subset='all'
):
    """
    Plot a set of posteriors with the original samples in turquoise and the reweighted posteriors in grey
    :param posterior: dictionary
        samples from a parameter estimation run
    :param new_weights: list
        list of weights, equal in length to the number of samples
    :param label: string
        string to use to label the output plot
    :param original_weights: list
        if needed, to denote original weights. useful if we need to get rid of certain samples
        (give them a weight of zero in both lists)_
    :param injection_values: dictionary
        optional dictionary of injected values
    """
    if original_weights is None:
        original_weights = [0.5] * len(new_weights)
    subset_keys = dict(
        all=utils.parameter_keys.keys(),
        intrinsic=['chirp_mass', 'mass_ratio', 'chi_eff', 'log_eccentricity'],
        extrinsic=['luminosity_distance', 'theta_jn', 'psi', 'phase']
    )
    posterior['log_eccentricity'] = -10
    posterior['log_eccentricity'][0] = -9
    parameters = [posterior[parameter] for parameter in subset_keys[subset]]
    labels = [utils.parameter_keys[parameter] for parameter in subset_keys[subset]]
    truths = None
    number_of_bins = 20
    bins, range = update_log_eccentricity_bins_and_range_for_corner(
                     -10, -8, number_of_bins, subset_keys, subset
                  )
    if injection_values is not None:
        truths = [injection_values[parameter] for parameter in subset_keys[subset]]
    figure = corner.corner(get_data(parameters), weights=original_weights, labels=labels,
                           bins=bins, smooth=0.9, label_kwargs=dict(fontsize=20), titles=True,
                           title_kwargs=dict(fontsize=20), color='darkturquoise', # color='#0072C1', 
                           range=range,
                           levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                           hist_kwargs=dict(density=True, lw=2),
                           plot_density=False, plot_datapoints=True, fill_contours=True, label='unweighted',
                           max_n_ticks=2)
    # Now add back in the real eccentricities
    posterior['log_eccentricity'] = np.log10(eccentricities)
    bins, range = update_log_eccentricity_bins_and_range_for_corner(
                     -5.68, np.log10(0.2), number_of_bins, subset_keys, subset
                  )
    parameters = [posterior[parameter] for parameter in subset_keys[subset]]
    corner.corner(get_data(parameters), weights=new_weights, bins=bins, labels=labels, fig=figure,
                  smooth=0.9, label_kwargs=dict(fontsize=20), titles=True,
                  title_kwargs=dict(fontsize=20), color='grey', truth_color='hotpink', #color='#228B22', truth_color='#FF8C00',
                  hist_kwargs=dict(density=True, lw=2), truths=truths, range=range,
                  levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                  plot_density=False, plot_datapoints=True, fill_contours=True, label='weighted',
                  max_n_ticks=2)
    for ax in figure.get_axes():
        ax.tick_params(axis='both', labelsize=18, labelrotation=25)
    plt.savefig("{}_{}_corner.pdf".format(label, subset))
    print('saving figure {}'.format("{}_{}_corner.pdf".format(label, subset)))
    plt.clf()


def plot_chi_p_histogram(result, output_folder, label=''):
    bin_number = 60
    align = 'left'
    rwidth = 0.999
    posterior = result.posterior
    priors = result.priors
    prior_samples = pd.DataFrame(priors.sample(len(posterior['luminosity_distance'])))
    posterior_with_spins = bb.gw.conversion.generate_all_bbh_parameters(posterior)
    priors_with_spins = bb.gw.conversion.generate_all_bbh_parameters(prior_samples)
    chi_p = posterior_with_spins['chi_p']
    chi_p_prior = priors_with_spins['chi_p']
    # Plot the chi_p histogram
    fig = plt.subplots(figsize=(6, 3.5))
    prior_n, _, _ = plt.hist(chi_p_prior, bins=bin_number, align=align, color='darkturquoise', rwidth=rwidth, alpha=0.3, label='prior', histtype='stepfilled')
    post_n, _, _ = plt.hist(chi_p, bins=bin_number, align=align, color='grey', rwidth=rwidth, alpha=0.75, label=label)
    #plt.axvline(0, color='hotpink', alpha=1, label='injection', linewidth=2)
    plt.legend(fontsize=14, loc='upper left')
    plt.yticks(visible=False)
    plt.grid(False)
    plt.xlabel('$\chi_{\\rm p}$', fontsize=14)
    plt.ylabel('$p(\chi_{\\rm p}|d)$', fontsize=14, labelpad=15)
    plt.xlim([0, 0.79])
    plt.ylim([0, np.max(post_n) + (np.max(post_n) / 3)])
    # Save the figure
    output_file = '{}/chi_p_histogram.pdf'.format(output_folder)
    print('output file: {}'.format(output_file))
    plt.savefig(output_file, bbox_inches='tight')


def plot_eccentricity_histogram(eccentricities, weights, injection, output_folder, label='', injected_eccentricity=0.1, minimum_log_eccentricity=-4, nbins=40):
    """
    Plot a 1d eccentricity histogram, with 90% credible interval bounds if injection=False,
    or an injected value line if injection=True.
    :param eccentricities: list
        list of eccentricities
    :param injection: boolean
        whether the data is taken from an injection
    :param output_folder: string
        location of the output (where we will save the figure)
    :param weights: list
        list of weights
    :pram injected_eccentricity: float
        the value of the injection, if injection is True
    """
    # Compute things to put on the plot
    maximum_eccentricity = wq.quantile(np.array(eccentricities), np.array(weights), 0.9)
    minimum_eccentricity = wq.quantile(np.array(eccentricities), np.array(weights), 0.1)
    print('e_max: {}, e_min: {}'.format(maximum_eccentricity, minimum_eccentricity))
    # Plot the eccentricity histogram
    fig = plt.subplots(figsize=(6, 3.5))
    n, bins, _ = plt.hist(eccentricities, bins=np.logspace(minimum_log_eccentricity, np.log10(0.2), nbins), align='left', color='white', rwidth=0.999,
             alpha=0, weights=weights, label=label)
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    # plot prior
    prior = []
    for bin in bins[:-1]:
        i = 0
        while i < int(len(eccentricities) * 5):
            prior.append(bin)
            i = i+1
    plt.hist(prior, bins=bins, align='left', color='darkturquoise', rwidth=0.999, alpha=0.3, label='prior', histtype='stepfilled')
    plt.hist(eccentricities, bins=np.logspace(minimum_log_eccentricity, np.log10(0.2), nbins), align='left', color='grey', rwidth=0.999,
             alpha=0.75, weights=weights, label=label)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    #if not injection:
        #plt.axvline(minimum_eccentricity, color='r', alpha=0.5)
        #plt.axvline(maximum_eccentricity, color='r', alpha=0.5)
    #else:
        #plt.axvline(injected_eccentricity, color='hotpink', alpha=1, label='injection', linewidth=2)
    plt.legend(fontsize=14)
    plt.yticks(visible=False)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.grid(False)
    plt.xscale('log')
    plt.xlabel('eccentricity, $e$', fontsize=14)
    plt.xlim([10**minimum_log_eccentricity, 0.2])
    plt.xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ['10$^{-6}$', '',  '10$^{-4}$', '', '10$^{-2}$', ''], ha='left', fontsize=12)
    plt.ylabel('$p(e|d)$', fontsize=14, labelpad=15)
    # Save the figure
    output_file = '{}/eccentricity_histogram.pdf'.format(output_folder)
    plt.savefig(output_file, bbox_inches='tight')
    return n, bins, _


def plot_2d_overlap(overlaps, time_grid_mesh, phase_grid_mesh):
    """
    Plot the 2D grid of time vs phase shifts.
    :param overlaps: 2D grid
        grid of calculated overlaps
    :param time_grid_mesh: 1D grid
        grid of proposed time shift
    :param phase_grid_mesh: 1D grid
        grid of proposed phase shifts
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    locator = ticker.LogLocator(base=np.exp(1))
    plt.contourf(phase_grid_mesh, np.divide(time_grid_mesh, 4096),
                 np.subtract(1, overlaps), locator=locator,
                 cmap='viridis_r')
    plt.ylabel("Time shift (s)")
    plt.xlabel("Phase shift")
    plt.colorbar()
    plt.title("1 - Overlap")
    plt.savefig("2d_overlap")
    plt.clf()


def plot_2d_heatmap_with_eccentricity(
        posterior_samples, parameter_name, weights, eccentricities, eccentricity_bins,
        injected_parameter=None, injected_eccentricity=None,
):
    """
    Plot a 2d heatmap of weighted posterior samples against eccentricity.
    :param posterior_samples: dictionary
        samples from a parameter estimation run
    :param parameter_name: string
        the name of the parameter, used to extract specific samples from the posterior samples
        and to label the plot
    :param injected_parameter: float
        value of the injected parameter
    :param injected_eccentricity: float
        value of the injected eccentricity
    :param weights: list
        list of weights for the posterior samples (will not be applied to the eccentricity data)
    :param eccentricities: list
        list of eccentricities
    :param eccentricity_bins: array
        bins into which to sort the eccentricities, i.e. np.logspace(-4, np.log10(0.2), 40)
    """
    number_of_bins = len(eccentricity_bins)
    hist_parameter, bin_edges_parameter = np.histogram(
        posterior_samples[parameter_name], bins=number_of_bins, weights=weights
    )
    hist_eccentricity, bin_edges_eccentricity = np.histogram(eccentricities, bins=eccentricity_bins)
    counts = [[p * e for e in hist_eccentricity] for p in hist_parameter]
    fig = plt.figure()
    plt.contourf(bin_edges_eccentricity[1:], bin_edges_parameter[1:], counts, cmap='Greys')
    plt.ylabel("reweighted {}".format(utils.parameter_keys[parameter_name]), fontsize=14)
    plt.xlabel("eccentricity, $e$", fontsize=14)
    if injected_eccentricity is not None:
        plt.axvline(injected_eccentricity, linewidth=1, color='darkturquoise', label='injection')
        plt.axhline(injected_parameter, linewidth=1, color='darkturquoise')
        plt.scatter(injected_eccentricity, injected_parameter, color='darkturquoise')
        plt.legend(fontsize=14)
    plt.savefig("heatmap_injection_{}.pdf".format(parameter_name), bbox_inches='tight')
    plt.clf()


def plot_detector_strain_asd(ax, df, interferometer):
    asd = bb.gw.utils.asd_from_freq_series(
        freq_data=interferometer.strain_data.frequency_domain_strain, df=df)
    ax.loglog(interferometer.strain_data.frequency_array[interferometer.strain_data.frequency_mask],
              asd[interferometer.strain_data.frequency_mask],
              label=interferometer.name)
    ax.loglog(interferometer.strain_data.frequency_array[interferometer.strain_data.frequency_mask],
              interferometer.amplitude_spectral_density_array[interferometer.strain_data.frequency_mask],
              lw=1.0, label=interferometer.name + ' ASD')


def plot_response_to_waveform(ax, df, waveform_polarisations, parameters, label, interferometer):
    response = interferometer.get_detector_response(waveform_polarisations, parameters)
    signal_asd = bb.gw.utils.asd_from_freq_series(
        freq_data=response, df=df)
    line, = ax.loglog(interferometer.strain_data.frequency_array[interferometer.strain_data.frequency_mask],
                      signal_asd[interferometer.strain_data.frequency_mask],
                      label=label)
    return line,


def _layout_frequency_domain_strain_plot(fig, ax):
    ax.grid(True)
    ax.set_ylabel(r'Strain [strain/$\sqrt{\rm Hz}$]')
    ax.set_xlabel(r'Frequency [Hz]')
    ax.legend(loc='best')
    fig.tight_layout()


def plot_detector_response(signals, parameters, labels, interferometer, show=True):
    fig, ax = plt.subplots()
    df = interferometer.strain_data.frequency_array[1] - interferometer.strain_data.frequency_array[0]
    plot_detector_strain_asd(ax, df, interferometer)
    for signal, label in zip(signals, labels):
        line, = plot_response_to_waveform(ax, df, signal, parameters, label, interferometer)
    _layout_frequency_domain_strain_plot(fig, ax)
    fig.savefig('{}_frequency_domain_data.png'.format(interferometer.name))
    if show:
        print(
            'displaying frequency domain strain for interferometer {} with labels {}'.format(
                interferometer.name, labels
            )
        )
        plt.show()
    plt.close(fig)


def animate_detector_response(
        eccentric_signal, circular_signal, eccentric_label, circular_label,
        eccentricities, parameters, interferometer,
        circular_waveform_generator, minimum_frequency, show=False,
):
    fig, ax = plt.subplots()
    df = interferometer.strain_data.frequency_array[1] - interferometer.strain_data.frequency_array[0]
    plot_detector_strain_asd(ax, df, interferometer)
    plot_response_to_waveform(ax, df, circular_signal, parameters, circular_label, interferometer)
    line, = plot_response_to_waveform(ax, df, eccentric_signal, parameters, eccentric_label, interferometer)
    _layout_frequency_domain_strain_plot(fig, ax)

    def init():
        line.set_ydata([np.nan]
                       * len(interferometer.strain_data.frequency_array[interferometer.strain_data.frequency_mask])
                       )
        return line,

    def animate(i):
        parameters['eccentricity'] = eccentricities[i]
        new_eccentric_waveform = wf.seobnre_bbh_with_spin_and_eccentricity(
            parameters, interferometer.sampling_frequency, minimum_frequency
        )
        eccentric_waveform, eccentric_signal, max_overlap, time_shift, phase_shift = ovlp.maximise_overlap(
            new_eccentric_waveform,
            circular_signal,
            circular_waveform_generator.sampling_frequency,
            circular_waveform_generator.frequency_array,
            interferometer.power_spectral_density,
            minimum_frequency=interferometer.minimum_frequency,
            maximum_frequency=interferometer.maximum_frequency
        )
        eccentric_response = interferometer.get_detector_response(eccentric_signal, parameters)
        seobnre_asd = bb.gw.utils.asd_from_freq_series(
            freq_data=eccentric_response, df=df)
        plt.title('{}: e={}, overlap={}'.format(interferometer.name, eccentricities[i], max_overlap))
        line.set_ydata(seobnre_asd[interferometer.strain_data.frequency_mask])  # update the data.
        return line,

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=1, blit=False, frames=len(eccentricities))
    ani.save('animate_{}.gif'.format(interferometer.name), writer=matplotlib.animation.PillowWriter(fps=10))
    if show:
        plt.show()
    plt.close(fig)

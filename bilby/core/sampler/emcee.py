from collections import namedtuple
import os
import signal
import shutil
from shutil import copyfile
import sys
import time

import numpy as np
from pandas import DataFrame
from distutils.version import LooseVersion
from emcee.autocorr import integrated_time
import dill as pickle

from ..utils import (
    logger, get_progress_bar, check_directory_exists_and_if_not_mkdir,
    latex_plot_format
)
from .base_sampler import MCMCSampler, SamplerError


class Emcee(MCMCSampler):
    """bilby wrapper emcee (https://github.com/dfm/emcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `emcee.EnsembleSampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    nwalkers: int, (100)
        The number of walkers
    nsteps: int, (100)
        The number of steps
    nburn: int (None)
        If given, the fixed number of steps to discard as burn-in. These will
        be discarded from the total number of steps set by `nsteps` and
        therefore the value must be greater than `nsteps`. Else, nburn is
        estimated from the autocorrelation time
    burn_in_fraction: float, (0.25)
        The fraction of steps to discard as burn-in in the event that the
        autocorrelation time cannot be calculated
    burn_in_act: float
        The number of autocorrelation times to discard as burn-in
    a: float (2)
        The proposal scale factor


    """

    default_kwargs = dict(
        nwalkers=500, a=2, args=[], kwargs={}, postargs=None, pool=None,
        live_dangerously=False, runtime_sortingfn=None, lnprob0=None,
        rstate0=None, blobs0=None, iterations=100, thin=1, storechain=True,
        mh_proposal=None, npool=1)

    def __init__(
            self,
            likelihood,
            priors,
            outdir='outdir',
            label='label',
            use_ratio=False,
            plot=False,
            skip_import_verification=False,
            pos0=None,
            nburn=None,
            burn_in_fraction=0.25,
            resume=True,
            burn_in_act=3,
            checkpoint_delta_t=600,
            **kwargs
    ):
        import emcee
        self.emcee = emcee

        if LooseVersion(emcee.__version__) > LooseVersion('2.2.1'):
            self.prerelease = True
        else:
            self.prerelease = False
        super(Emcee, self).__init__(
            likelihood=likelihood, priors=priors, outdir=outdir,
            label=label, use_ratio=use_ratio, plot=plot,
            skip_import_verification=skip_import_verification, **kwargs)
        self.emcee = self._check_version()
        self.resume = resume
        self.pos0 = pos0
        self.nburn = nburn
        self.burn_in_fraction = burn_in_fraction
        self.burn_in_act = burn_in_act
        self.tau_list = list()
        self.checkpoint_delta_t = checkpoint_delta_t

        signal.signal(signal.SIGTERM, self.checkpoint_and_exit)
        signal.signal(signal.SIGINT, self.checkpoint_and_exit)

    def _check_version(self):
        import emcee
        if LooseVersion(emcee.__version__) > LooseVersion('2.2.1'):
            self.prerelease = True
        else:
            self.prerelease = False
        return emcee

    def _translate_kwargs(self, kwargs):
        if 'nwalkers' not in kwargs:
            for equiv in self.nwalkers_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nwalkers'] = kwargs.pop(equiv)
        if 'iterations' not in kwargs:
            if 'nsteps' in kwargs:
                kwargs['iterations'] = kwargs.pop('nsteps')

    @property
    def sampler_function_kwargs(self):
        keys = ['lnprob0', 'rstate0', 'blobs0', 'iterations', 'thin',
                'storechain', 'mh_proposal']

        # updated function keywords for emcee > v2.2.1
        updatekeys = {'p0': 'initial_state',
                      'lnprob0': 'log_prob0',
                      'storechain': 'store'}

        function_kwargs = {key: self.kwargs[key] for key in keys if key in self.kwargs}
        function_kwargs['p0'] = self.pos0

        if self.prerelease:
            if function_kwargs['mh_proposal'] is not None:
                logger.warning("The 'mh_proposal' option is no longer used "
                               "in emcee v{}, and will be ignored.".format(
                                   self.emcee.__version__))
            del function_kwargs['mh_proposal']

            for key in updatekeys:
                if updatekeys[key] not in function_kwargs:
                    function_kwargs[updatekeys[key]] = function_kwargs.pop(key)
                else:
                    del function_kwargs[key]

        return function_kwargs

    @property
    def sampler_init_kwargs(self):
        init_kwargs = {key: value
                       for key, value in self.kwargs.items()
                       if key not in self.sampler_function_kwargs}

        init_kwargs['lnpostfn'] = _log_posterior
        init_kwargs['dim'] = self.ndim

        # updated init keywords for emcee > v2.2.1
        updatekeys = {'dim': 'ndim',
                      'lnpostfn': 'log_prob_fn'}

        if self.prerelease:
            for key in updatekeys:
                if key in init_kwargs:
                    init_kwargs[updatekeys[key]] = init_kwargs.pop(key)

            oldfunckeys = ['p0', 'lnprob0', 'storechain', 'mh_proposal']
            for key in oldfunckeys:
                if key in init_kwargs:
                    del init_kwargs[key]

        init_kwargs["pool"] = None
        if "npool" in init_kwargs:
            del init_kwargs["npool"]

        from emcee import moves
        init_kwargs["moves"] = [
            [moves.StretchMove(), 25],
            [moves.WalkMove(), 10],
            [moves.DEMove(), 10],
            [moves.DESnookerMove(), 5],
            # [moves.GaussianMove(cov=1), 2],
            [moves.KDEMove(), 25],
        ]

        return init_kwargs

    def lnpostfn(self, theta):
        log_prior = self.log_prior(theta)
        if np.isinf(log_prior):
            return -np.inf, [np.nan, np.nan]
        else:
            log_likelihood = self.log_likelihood(theta)
            return log_likelihood + log_prior, [log_likelihood, log_prior]

    @property
    def ln_post_chain(self):
        return np.sum(self.sampler.blobs, axis=-1)

    @property
    def nburn(self):
        if type(self.__nburn) in [float, int]:
            return int(self.__nburn)
        elif self.__nburn == "ln_post":
            return running_mean_burn_in(self.ln_post_chain, window=20, threshold=0.5)
        elif self.result.max_autocorrelation_time is None:
            return int(self.burn_in_fraction * self.nsteps)
        else:
            return int(self.burn_in_act * self.result.max_autocorrelation_time)

    @nburn.setter
    def nburn(self, nburn):
        if isinstance(nburn, (float, int)):
            if nburn > self.kwargs['iterations'] - 1:
                raise ValueError('Number of burn-in samples must be smaller '
                                 'than the total number of iterations')

        self.__nburn = nburn

    @property
    def nwalkers(self):
        return self.kwargs['nwalkers']

    @property
    def nsteps(self):
        return self.kwargs['iterations']

    @nsteps.setter
    def nsteps(self, nsteps):
        self.kwargs['iterations'] = nsteps

    @property
    def stored_chain(self):
        """ Read the stored zero-temperature chain data in from disk """
        return np.genfromtxt(self.checkpoint_info.chain_file, names=True)

    @property
    def stored_samples(self):
        """ Returns the samples stored on disk """
        return self.stored_chain[self.search_parameter_keys]

    @property
    def stored_loglike(self):
        """ Returns the log-likelihood stored on disk """
        return self.stored_chain['log_l']

    @property
    def stored_logprior(self):
        """ Returns the log-prior stored on disk """
        return self.stored_chain['log_p']

    def _init_chain_file(self):
        with open(self.checkpoint_info.chain_file, "w+") as ff:
            ff.write('walker\t{}\tlog_l\tlog_p\n'.format(
                '\t'.join(self.search_parameter_keys)))

    @property
    def checkpoint_info(self):
        """ Defines various things related to checkpointing and storing data

        Returns
        -------
        checkpoint_info: named_tuple
            An object with attributes `sampler_file`, `chain_file`, and
            `chain_template`. The first two give paths to where the sampler and
            chain data is stored, the last a formatted-str-template with which
            to write the chain data to disk

        """
        out_dir = os.path.join(
            self.outdir, '{}_{}'.format(self.__class__.__name__.lower(),
                                        self.label))
        check_directory_exists_and_if_not_mkdir(out_dir)

        chain_file = os.path.join(out_dir, 'chain.dat')
        sampler_file = os.path.join(out_dir, 'sampler.pickle')
        chain_template =\
            '{:d}' + '\t{:.9e}' * (len(self.search_parameter_keys) + 2) + '\n'

        CheckpointInfo = namedtuple(
            'CheckpointInfo', ['sampler_file', 'chain_file', 'chain_template'])

        checkpoint_info = CheckpointInfo(
            sampler_file=sampler_file, chain_file=chain_file,
            chain_template=chain_template)

        return checkpoint_info

    @property
    def sampler_chain(self):
        nsteps = self._previous_iterations
        return self.sampler.chain[:, :nsteps, :]

    def checkpoint(self):
        """ Writes a pickle file of the sampler to disk using dill """
        logger.info("Checkpointing sampler to file {}"
                    .format(self.checkpoint_info.sampler_file))
        with open(self.checkpoint_info.sampler_file, 'wb') as f:
            # Overwrites the stored sampler chain with one that is truncated
            # to only the completed steps
            self._sampler._chain = self.sampler_chain
            self._sampler.tau_list = self.tau_list
            self._sampler.pool = None
            pickle.dump(self._sampler, f)
        self._sampler.pool = self.pool

    def checkpoint_and_exit(self, signum, frame):
        logger.info("Recieved signal {}".format(signum))
        self.checkpoint()
        self._close_pool()
        sys.exit()

    def _initialise_sampler(self):
        self._sampler = self.emcee.EnsembleSampler(**self.sampler_init_kwargs)
        self._sampler.pool = self.pool
        self._init_chain_file()

    def _setup_pool(self):
        if self.kwargs["pool"] is not None:
            logger.info("Using user defined pool.")
            self.pool = self.kwargs["pool"]
        elif self.npool > 1:
            logger.info(
                "Setting up multiproccesing pool with {} processes.".format(
                    self.npool
                )
            )
            import multiprocessing
            self.pool = multiprocessing.Pool(
                processes=self.npool,
                initializer=_initialize_global_variables,
                initargs=(
                    self.likelihood,
                    self.priors,
                    self._search_parameter_keys,
                    self.use_ratio
                )
            )
        else:
            self.pool = None
        _initialize_global_variables(
            likelihood=self.likelihood,
            priors=self.priors,
            search_parameter_keys=self._search_parameter_keys,
            use_ratio=self.use_ratio
        )
        self.kwargs["pool"] = None

    def _close_pool(self):
        if getattr(self, "pool", None) is not None:
            logger.info("Starting to close worker pool.")
            self.pool.close()
            self.pool.join()
            self._sampler.pool = None
            self.pool = None
            self.kwargs["pool"] = self.pool
            logger.info("Finished closing worker pool.")

    @property
    def sampler(self):
        """ Returns the ptemcee sampler object

        If, alrady initialized, returns the stored _sampler value. Otherwise,
        first checks if there is a pickle file from which to load. If there is
        not, then initialize the sampler and set the initial random draw

        """
        if hasattr(self, '_sampler'):
            pass
        elif self.resume and os.path.isfile(self.checkpoint_info.sampler_file):
            logger.info("Resuming run from checkpoint file {}"
                        .format(self.checkpoint_info.sampler_file))
            with open(self.checkpoint_info.sampler_file, 'rb') as f:
                self._sampler = pickle.load(f)
                self.tau_list = self._sampler.tau_list
                del self._sampler.tau_list
                self._sampler.pool = self.pool
            self._set_pos0_for_resume()
        else:
            self._initialise_sampler()
            self._set_pos0()
        return self._sampler

    def write_chains_to_file(self, sample):
        chain_file = self.checkpoint_info.chain_file
        temp_chain_file = chain_file + '.temp'
        if os.path.isfile(chain_file):
            copyfile(chain_file, temp_chain_file)
        if self.prerelease:
            points = np.hstack([sample.coords, sample.blobs])
        else:
            points = np.hstack([sample[0], np.array(sample[3])])
        with open(temp_chain_file, "a") as ff:
            for ii, point in enumerate(points):
                ff.write(self.checkpoint_info.chain_template.format(ii, *point))
        shutil.move(temp_chain_file, chain_file)

    @property
    def _previous_iterations(self):
        """ Returns the number of iterations that the sampler has saved

        This is used when loading in a sampler from a pickle file to figure out
        how much of the run has already been completed
        """
        try:
            return len(self.sampler.blobs)
        except AttributeError:
            return 0

    def _draw_pos0_from_prior(self):
        _, points, ln_l = self.get_initial_points_from_prior(self.nwalkers, ln_l_min=-np.inf)
        return points

    @property
    def _pos0_shape(self):
        return (self.nwalkers, self.ndim)

    def _set_pos0(self):
        if self.pos0 is not None:
            logger.debug("Using given initial positions for walkers")
            if isinstance(self.pos0, DataFrame):
                self.pos0 = self.pos0[self.search_parameter_keys].values
            elif type(self.pos0) in (list, np.ndarray):
                self.pos0 = np.squeeze(self.pos0)

            if self.pos0.shape != self._pos0_shape:
                raise ValueError(
                    'Input pos0 should be of shape ndim, nwalkers')
            logger.debug("Checking input pos0")
            for draw in self.pos0:
                self.check_draw(draw)
        else:
            logger.debug("Generating initial walker positions from prior")
            self.pos0 = self._draw_pos0_from_prior()

    def _set_pos0_for_resume(self):
        self.pos0 = self.sampler.chain[:, -1, :]

    def _make_plots(self):
        plot_ln_post(
            array=self.ln_post_chain,
            outdir=self.outdir,
            label=self.label,
            window=20,
            threshold=0.5
        )
        plot_tau(
            tau_list_n=np.arange(len(self.tau_list)) + 1,
            tau_list=self.tau_list,
            search_parameter_keys=self.search_parameter_keys,
            outdir=self.outdir,
            label=self.label,
            tau=1,
            autocorr_tau=1
        )
        thin = getattr(self.result, "max_autocorrelation_time", 1)
        if thin is None:
            thin = 1
        plot_walkers(
            walkers=self.sampler_chain,
            nburn=self.nburn,
            thin=thin,
            parameter_labels=self.search_parameter_keys,
            outdir=self.outdir,
            label=self.label
        )

    def run_sampler(self):
        self._setup_pool()
        tqdm = get_progress_bar()
        sampler_function_kwargs = self.sampler_function_kwargs
        iterations = sampler_function_kwargs.pop('iterations')
        iterations -= self._previous_iterations

        if self.prerelease:
            sampler_function_kwargs['initial_state'] = self.pos0
        else:
            sampler_function_kwargs['p0'] = self.pos0

        last_checkpoint_time = time.time()
        for sample in tqdm(
                self.sampler.sample(iterations=iterations, **sampler_function_kwargs),
                total=iterations):
            sample.coords[:, _periodic] = np.mod(
                sample.coords[:, _periodic] - _minima[_periodic], _range[_periodic]
            ) + _minima[_periodic]
            self.sampler.chain[:, :, _periodic] = np.mod(
                self.sampler.chain[:, :, _periodic] - _minima[_periodic], _range[_periodic]
            ) + _minima[_periodic]
            self.tau_list.append(integrated_time(
                np.swapaxes(self.sampler.chain, 0, 1), tol=0
            ))
            if time.time() - last_checkpoint_time > self.checkpoint_delta_t:
                self.checkpoint()
                self._make_plots()
                last_checkpoint_time = time.time()
        self._close_pool()
        self.checkpoint()

        self.result.sampler_output = np.nan
        self.calculate_autocorrelation(np.swapaxes(self.sampler.chain, 0, 1)[self.nburn:])
        self._make_plots()

        self.print_nburn_logging_info()

        self._generate_result()

        self.result.samples = self.sampler.chain[:, self.nburn:, :].reshape(
            (-1, self.ndim))
        self.result.walkers = self.sampler.chain
        return self.result

    def _generate_result(self):
        self.result.nburn = self.nburn
        self.calc_likelihood_count()
        if self.result.nburn > self.nsteps:
            raise SamplerError(
                "The run has finished, but the chain is not burned in: "
                "`nburn < nsteps` ({} < {}). Try increasing the "
                "number of steps.".format(self.result.nburn, self.nsteps))
        blobs = np.array(self.sampler.blobs)
        blobs_trimmed = blobs[self.nburn:, :, :].reshape((-1, 2))
        log_likelihoods, log_priors = blobs_trimmed.T
        self.result.log_likelihood_evaluations = log_likelihoods
        self.result.log_prior_evaluations = log_priors
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan


def running_mean_burn_in(array, window=20, threshold=0.5):
    mu_on_sigma = running_mean_on_sigma(np.mean(array, axis=1))
    if len(mu_on_sigma) <= 2 * window:
        return len(mu_on_sigma) + 1
    elif np.all(mu_on_sigma[:-window] >= threshold):
        return len(mu_on_sigma) + 1
    elif np.all(mu_on_sigma[:-window] <= threshold):
        return 0
    else:
        return min(
            len(mu_on_sigma) + 1,
            np.where(mu_on_sigma[:-window] >= threshold)[0][-1] + window
        )


def running_mean_on_sigma(array, window=20):
    array_gradient = np.gradient(array)
    running_mean = np.convolve(array_gradient, np.ones((window,)) / window)[(window - 1):]
    running_sigma = (
        np.convolve(array_gradient ** 2, np.ones((window,)) / window)[(window - 1):]
        - running_mean ** 2
    ) ** 0.5
    return running_mean / running_sigma


@latex_plot_format
def plot_ln_post(array, outdir, label, window=20, threshold=0.5):
    import matplotlib.pyplot as plt
    mean_array = np.mean(array, axis=1)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 15), sharex=True)
    axes[0].plot(mean_array, color="C0")
    axes[0].plot(np.min(array, axis=1), color="C0", linestyle="--")
    axes[0].plot(np.max(array, axis=1), color="C0", linestyle="--")
    axes[0].fill_between(
        x=np.arange(len(mean_array)),
        y1=np.min(array, axis=1),
        y2=np.max(array, axis=1),
        alpha=0.2,
        color="C0"
    )
    axes[1].plot(np.log(np.std(array, axis=1)))
    mu_on_sigma = running_mean_on_sigma(array=mean_array, window=window)
    mu_on_sigma_2 = running_mean_on_sigma(array=np.std(array, axis=1), window=window)
    axes[2].plot(mu_on_sigma)
    axes[2].plot(mu_on_sigma_2)
    axes[2].fill_between(
        x=np.arange(len(mu_on_sigma)),
        y1=-threshold * np.ones_like(mu_on_sigma),
        y2=threshold * np.ones_like(mu_on_sigma),
        alpha=0.2,
        color="g"
    )
    axes[2].set_xlabel("Iteration")
    axes[0].set_ylabel("$\\langle \\ln P \\rangle$")
    axes[1].set_ylabel("$\\Delta \\langle \\ln P \\rangle$")
    axes[2].set_ylabel("Running mean / sigma")
    axes[0].set_ylim(np.min(mean_array), 1.2 * np.max(mean_array) - 0.2 * np.min(mean_array))
    axes[2].set_xlim(0, len(mu_on_sigma) - 1)
    plt.savefig(f"{outdir}/{label}_ln_likelihood.png")
    plt.tight_layout()
    plt.close(fig)


@latex_plot_format
def plot_tau(
        tau_list_n, tau_list, search_parameter_keys, outdir, label, tau, autocorr_tau
):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i, key in enumerate(search_parameter_keys):
        ax.plot(tau_list_n, np.array(tau_list)[:, i], label=key.replace("_", " "))
    ax.axvline(tau_list_n[-1] - tau * autocorr_tau)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\langle \tau \rangle$")
    ax.legend()
    fig.savefig("{}/{}_checkpoint_tau.png".format(outdir, label))
    plt.tight_layout()
    plt.close(fig)


@latex_plot_format
def plot_walkers(walkers, nburn, thin, parameter_labels, outdir, label):
    """ Method to plot the trace of the walkers in an ensemble MCMC plot """
    import matplotlib.pyplot as plt
    nwalkers, nsteps, ndim = walkers.shape
    if nsteps > 100:
        thin = 5
    if nsteps > 1000:
        thin = 50
    idxs = np.arange(nsteps)
    fig, axes = plt.subplots(nrows=ndim, ncols=2, figsize=(8, 3 * ndim))
    scatter_kwargs = dict(lw=0, marker="o", markersize=1, alpha=0.05,)
    # Plot the burn-in
    for i, (ax, axh) in enumerate(axes):
        ax.plot(
            idxs[:nburn + 1:thin],
            walkers[:, :nburn + 1:thin, i].T,
            color="C1",
            **scatter_kwargs
        )
        if nburn >= len(idxs):
            axh.hist(
                walkers[:, -1, i], bins=nwalkers // 10, alpha=0.8,
                color="C1", histtype="stepfilled", density=True
            )

    # Plot the thinned posterior samples
    for i, (ax, axh) in enumerate(axes):
        ax.plot(
            idxs[nburn + 1::thin],
            walkers[:, nburn + 1::thin, i].T,
            color="C0",
            **scatter_kwargs
        )
        for jj in np.arange(nburn + 1, len(idxs), thin):
            axh.hist(
                walkers[:, jj, i], bins=nwalkers // 10,
                histtype="stepfilled", density=True, color="C0",
                alpha=min(thin / (len(idxs) - nburn - 1), 1)
            )
        axh.set_xlabel(parameter_labels[i].replace("_", " "))
        ax.set_ylabel(parameter_labels[i].replace("_", " "))

    fig.tight_layout()
    filename = "{}/{}_checkpoint_trace.png".format(outdir, label)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _initialize_global_variables(
        likelihood, priors, search_parameter_keys, use_ratio
):
    """
    Store a global copy of the likelihood, priors, and search keys for
    multiprocessing.
    """
    global _likelihood
    global _priors
    global _periodic
    global _search_parameter_keys
    global _use_ratio
    global _minima
    global _range
    _likelihood = likelihood
    _priors = priors
    _search_parameter_keys = search_parameter_keys
    _use_ratio = use_ratio
    _periodic = [
        priors[key].boundary == "periodic" for key in search_parameter_keys
    ]
    _minima = np.array([
        priors[key].minimum for key in search_parameter_keys
    ])
    _range = np.array([
        priors[key].maximum for key in search_parameter_keys
    ]) - _minima


_likelihood = None
_priors = None
_periodic = None
_search_parameter_keys = None
_use_ratio = None
_minima = None
_range = None


def _log_posterior(theta):
    theta[_periodic] = np.mod(
        theta[_periodic] - _minima[_periodic], _range[_periodic]
    ) + _minima[_periodic]
    params = {
        key: t for key, t in zip(_search_parameter_keys, theta)
    }
    log_prior = _priors.ln_prob(params)
    if np.isinf(log_prior):
        return -np.inf, [np.nan, np.nan]
    else:
        _likelihood.parameters.update(params)
        if _use_ratio:
            log_likelihood = _likelihood.log_likelihood_ratio()
        else:
            log_likelihood = _likelihood.log_likelihood()
    return log_likelihood + log_prior, [log_likelihood, log_prior]

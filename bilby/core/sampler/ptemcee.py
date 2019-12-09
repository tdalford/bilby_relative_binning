from __future__ import absolute_import, division, print_function

import os
from shutil import copyfile
import signal
import sys

import numpy as np
from pandas import DataFrame

from ..utils import logger, get_progress_bar, plot_walkers
from .emcee import Emcee
from .base_sampler import SamplerError


class Ptemcee(Emcee):
    """bilby wrapper ptemcee (https://github.com/willvousden/ptemcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `ptemcee.Sampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    nwalkers: int, (100)
        The number of walkers
    nsteps: int, (100)
        The number of steps to take
    nburn: int (50)
        The fixed number of steps to discard as burn-in
    ntemps: int (2)
        The number of temperatures used by ptemcee

    """
    default_kwargs = dict(
        ntemps=3, nwalkers=100, Tmax=None, betas=None, log10betamin=None,
        threads=1, pool=None, a=2.0, loglargs=[], logpargs=[], loglkwargs={},
        logpkwargs={}, adaptation_lag=10000, adaptation_time=100, random=None,
        iterations=10000, storechain=True, adapt=True, swap_ratios=False,
        n_check=10, n_check_initial=50, n_effective=500, pos0=None)

    def __init__(self, likelihood, priors, outdir='outdir', label='label',
                 use_ratio=False, plot=False, skip_import_verification=False,
                 nburn=None, burn_in_fraction=0.25, burn_in_act=3, resume=True,
                 **kwargs):
        super(Ptemcee, self).__init__(
            likelihood=likelihood, priors=priors, outdir=outdir,
            label=label, use_ratio=use_ratio, plot=plot,
            skip_import_verification=skip_import_verification,
            nburn=nburn, burn_in_fraction=burn_in_fraction,
            burn_in_act=burn_in_act, resume=resume, **kwargs)

        signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
        signal.signal(signal.SIGINT, self.write_current_state_and_exit)
        signal.signal(signal.SIGALRM, self.write_current_state_and_exit)

    @property
    def internal_kwargs(self):
        keys = ["n_check", "n_effective", "n_check_initial", "pos0"]
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler_function_kwargs(self):
        keys = ['iterations', 'storechain', 'adapt', 'swap_ratios']
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler_init_kwargs(self):
        return {key: value
                for key, value in self.kwargs.items()
                if key not in self.internal_kwargs and
                key not in self.sampler_function_kwargs}

    @property
    def ntemps(self):
        return self.kwargs['ntemps']

    def print_tswap_acceptance_fraction(self):
        logger.info("Sampler per-chain tswap acceptance fraction = {}".format(
            self.sampler.tswap_acceptance_fraction))

    def write_chains_to_file(self):
        chain_file = self.checkpoint_info['chain_file']
        np.savez(
            chain_file,
            pos_array=np.array(self.pos_list),
            loglike_array=np.array(self.loglike_list),
            logpost_array=np.array(self.logpost_list))

    def read_chains_from_file(self):
        chain_file = self.checkpoint_info['chain_file']
        data = np.load(chain_file)

        self.pos_list = list(data.get("pos_array"))
        self.loglike_list = list(data.get("loglike_array"))
        self.logpost_list = list(data.get("logpost_array"))
        logger.info("Found {} iterations on file".format(len(self.pos_list)))

    def initialise_lists(self):
        chain_file = self.checkpoint_info["chain_file"]
        if os.path.isfile(chain_file) and self.resume:
            logger.info("Reading in resume data from {}".format(chain_file))
            self.read_chains_from_file()
        else:
            self.pos_list = []
            self.loglike_list = []
            self.logpost_list = []

    def write_current_state_and_exit(self, signum=None, frame=None):
        self.pbar.close()
        logger.warning("Run terminated with signal {}".format(signum))
        sys.exit(130)

    def _draw_pos0_from_prior(self):
        # for ptemcee, the pos0 has the shape ntemps, nwalkers, ndim
        return [[self.get_random_draw_from_prior()
                 for _ in range(self.nwalkers)]
                for _ in range(self.kwargs['ntemps'])]

    @property
    def _pos0_shape(self):
        return (self.ntemps, self.nwalkers, self.ndim)

    def check_n_effective(self, ii):
        #samples_so_far = self.sampler.chain.reshape((-1, self.ndim))[:ii, :]
        samples_so_far = np.array(self.pos_list)[:, 0, :, :].reshape((-1, self.ndim))
        self.calculate_autocorrelation(samples_so_far)

        if (self.result.max_autocorrelation_time is None or self.result.max_autocorrelation_time == 0):
            logger.debug("Unable to calculate max autocorrelation time")
            return False
        if ii < self.nburn:
            logger.debug("ii={} < nburn={}".format(ii, self.nburn))
            return False

        self.result.n_effective = np.max([0, int(
            0.5 * (ii - self.nburn) * self.nwalkers / self.result.max_autocorrelation_time)])
        return self.result.n_effective > self.internal_kwargs["n_effective"]

    def print_func(self, niter):
        string = []
        string.append("nburn:{:d}".format(self.nburn))
        string.append("max_act:{}".format(self.result.max_autocorrelation_time))
        n_eff = getattr(self.result, 'n_effective', None)
        string.append("neff:{}/{}".format(
            n_eff, self.internal_kwargs["n_effective"]))

        self.pbar.set_postfix_str(" ".join(string), refresh=False)
        self.pbar.update()

    def run_sampler(self):
        sampler_function_kwargs = self.sampler_function_kwargs
        iterations = sampler_function_kwargs.pop('iterations')

        sampler = self.sampler
        self.initialise_lists()
        pos0 = self.get_pos0()

        tqdm = get_progress_bar()
        self.pbar = tqdm(file=sys.stdout, initial=len(self.pos_list))

        # main iteration loop
        n_success = 0
        for ii, _ in enumerate(sampler.sample(pos0, iterations=iterations,
                                              **sampler_function_kwargs)):
            self.print_func(ii)
            self.pos_list.append(sampler.chain[:, :, ii, :])
            self.logpost_list.append(sampler.logprobability[:, :, ii])
            self.loglike_list.append(sampler.loglikelihood[:, :, ii])
            self.write_chains_to_file()
            if (ii > self.internal_kwargs["n_check_initial"] and
                ii % self.internal_kwargs["n_check"] == 0):
                if self.check_n_effective(ii):
                    n_success += 1
                self.plot_traceplot()
            logger.debug("N success={}".format(n_success))
            if n_success >= 2:
                logger.info(
                    "Stopping sampling on iteration {}/{} as n_effective>{}"
                    .format(ii, iterations, self.internal_kwargs["n_effective"]))
                self.nsteps = ii
                break
        self.pbar.close()

        self.result.sampler_output = np.nan
        self.print_nburn_logging_info()
        self.print_tswap_acceptance_fraction()

        self.result.nburn = self.nburn
        if self.result.nburn > self.nsteps:
            raise SamplerError(
                "The run has finished, but the chain is not burned in: "
                "`nburn={} < nsteps={}`. Try increasing the number of steps or the "
                "number of effective samples".format(self.result.nburn, self.nsteps))
        self.calc_likelihood_count()

        pos_array = np.array(self.pos_list)
        loglike_array = np.array(self.loglike_list)
        logpost_array = np.array(self.logpost_list)

        self.result.samples = pos_array[self.nburn:, 0, :, :].reshape(
            (-1, self.ndim))
        self.result.walkers = np.swapaxes(pos_array[:, 0, :, :], 0, 1)

        self.result.log_likelihood_evaluations = (
            loglike_array[self.nburn:, 0, :].reshape((-1)))
        self.result.log_prior_evaluations = (
            logpost_array - loglike_array)[self.nburn:, 0, :].reshape((-1))
        self.result.betas = self.sampler.betas
        self.result.log_evidence, self.result.log_evidence_err =\
            self.sampler.log_evidence_estimate(
                self.sampler.loglikelihood, self.nburn / self.nsteps)

        return self.result

    def plot_traceplot(self):
        label = self.label + "_traceplot"
        walkers = np.swapaxes(np.array(self.pos_list)[:, 0, :, :], 0, 1)
        plot_walkers(walkers, parameter_labels=self.search_parameter_keys,
                     nburn=self.nburn, outdir=self.outdir, label=label)

    @property
    def sampler(self):
        """ Returns the ptemcee sampler object

        If, alrady initialized, returns the stored _sampler value. Otherwise,
        first checks if there is a pickle file from which to load. If there is
        not, then initialize the sampler and set the initial random draw

        """
        if hasattr(self, '_sampler'):
            pass
        else:
            import ptemcee

            kwargs = self.sampler_init_kwargs
            if kwargs["betas"] is None:
                if kwargs["log10betamin"] is not None:
                    betas = np.logspace(0, kwargs["log10betamin"], self.ntemps)
                    kwargs["betas"] = betas
                    logger.info("Using betas={}".format(betas))
            kwargs.pop("log10betamin")

            self._sampler = ptemcee.Sampler(
                dim=self.ndim, logl=self.log_likelihood, logp=self.log_prior,
                **kwargs)
        return self._sampler

    @property
    def checkpoint_info(self):
        """ 
        """

        out_dir = self.checkpoint_outdir
        chain_file = os.path.join(out_dir, 'chain.npz')
        traceplot = os.path.join(self.outdir, '{}_traceplot.png'.format(self.label))
        return dict(chain_file=chain_file, traceplot=traceplot)

    def get_pos0(self):
        if self.kwargs.get('pos0', None) is not None:
            logger.debug("Using given initial positions for walkers")
            if isinstance(self.pos0, DataFrame):
                pos0 = self.pos0[self.search_parameter_keys].values
            elif type(self.pos0) in (list, np.ndarray):
                pos0 = np.squeeze(self.pos0)

            if pos0.shape != self._pos0_shape:
                raise ValueError(
                    'Input pos0 should be of shape ndim, nwalkers')
            logger.debug("Checking input pos0")
            for draw in pos0:
                self.check_draw(draw)
        elif len(self.pos_list) > 0:
            pos0 = self.pos_list[-1]
        else:
            logger.debug("Generating initial walker positions from prior")
            pos0 = self._draw_pos0_from_prior()

        return pos0
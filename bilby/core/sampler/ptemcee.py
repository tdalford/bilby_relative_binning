from __future__ import absolute_import, division, print_function

import os
from shutil import copyfile
import signal
import sys

import numpy as np

from ..utils import logger, get_progress_bar
from . import Emcee
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
        ntemps=3, nwalkers=100, Tmax=None, betas=None, threads=1, pool=None,
        a=2.0, loglargs=[], logpargs=[], loglkwargs={}, logpkwargs={},
        adaptation_lag=10000, adaptation_time=100, random=None, iterations=1000,
        storechain=True, adapt=True, swap_ratios=False,
        n_check=10, n_check_initial=50, n_effective=500)

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
        keys = ["n_check", "n_effective", "n_check_initial"]
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

    @property
    def sampler_chain(self):
        nsteps = self._previous_iterations
        return self.sampler.chain[:, :, :nsteps, :]

    def _initialise_sampler(self):
        import ptemcee
        self._sampler = ptemcee.Sampler(
            dim=self.ndim, logl=self.log_likelihood, logp=self.log_prior,
            **self.sampler_init_kwargs)
        self._init_chain_file()

    def print_tswap_acceptance_fraction(self):
        logger.info("Sampler per-chain tswap acceptance fraction = {}".format(
            self.sampler.tswap_acceptance_fraction))

    def write_chains_to_file(self, pos, loglike, logpost):
        chain_file = self.checkpoint_info.chain_file
        temp_chain_file = chain_file + '.temp'
        if os.path.isfile(chain_file):
            try:
                copyfile(chain_file, temp_chain_file)
            except OSError:
                logger.warning("Failed to write temporary chain file {}".format(temp_chain_file))

        with open(temp_chain_file, "a") as ff:
            loglike = np.squeeze(loglike[0, :])
            logprior = np.squeeze(logpost[0, :]) - loglike
            for ii, (point, logl, logp) in enumerate(zip(pos[0, :, :], loglike, logprior)):
                line = np.concatenate((point, [logl, logp]))
                ff.write(self.checkpoint_info.chain_template.format(ii, *line))
        os.rename(temp_chain_file, chain_file)

    def write_current_state_and_exit(self, signum=None, frame=None):
        logger.warning("Run terminated with signal {}".format(signum))
        sys.exit(130)

    @property
    def _previous_iterations(self):
        """ Returns the number of iterations that the sampler has saved

        This is used when loading in a sampler from a pickle file to figure out
        how much of the run has already been completed
        """
        return self.sampler.time

    def _draw_pos0_from_prior(self):
        # for ptemcee, the pos0 has the shape ntemps, nwalkers, ndim
        return [[self.get_random_draw_from_prior()
                 for _ in range(self.nwalkers)]
                for _ in range(self.kwargs['ntemps'])]

    @property
    def _pos0_shape(self):
        return (self.ntemps, self.nwalkers, self.ndim)

    def _set_pos0_for_resume(self):
        self.pos0 = None

    def check_n_effective(self, ii):
        self.calculate_autocorrelation(
            self.sampler.chain.reshape((-1, self.ndim)))

        if (self.result.max_autocorrelation_time is None or self.result.max_autocorrelation_time == 0):
            logger.debug("Unable to calculate max autocorrelation time")
            return False
        if ii < self.nburn:
            logger.debug("ii={} < nburn={}".format(ii, self.nburn))
            return False

        self.result.n_effective = np.max([0, int(
            ii * self.nwalkers / self.result.max_autocorrelation_time) - self.nburn])
        return self.result.n_effective > self.internal_kwargs["n_effective"]

    def print_func(self, niter):
        string = []
        string.append("nburn:{:d}".format(self.nburn))
        string.append("max_act:{}".format(self.result.max_autocorrelation_time))
        n_eff = getattr(self.result, 'n_effective', 0)
        string.append("neff:{}/{}".format(
            n_eff, self.internal_kwargs["n_effective"]))

        self.pbar.set_postfix_str(" ".join(string), refresh=False)
        self.pbar.update(niter - self.pbar.n)

    def run_sampler(self):
        sampler_function_kwargs = self.sampler_function_kwargs
        iterations = sampler_function_kwargs.pop('iterations')
        iterations -= self._previous_iterations

        tqdm = get_progress_bar()
        self.pbar = tqdm(file=sys.stdout, total=iterations)

        # main iteration loop
        n_success = 0
        for ii, (pos, logpost, loglike) in enumerate(self.sampler.sample(self.pos0, iterations=iterations, **sampler_function_kwargs)):
            self.write_chains_to_file(pos, loglike, logpost)
            self.print_func(ii)
            if (ii > self.internal_kwargs["n_check_initial"] and
                ii % self.internal_kwargs["n_check"] == 0 and
                self.check_n_effective(ii)):
                n_success += 1
            logger.debug("N success={}".format(n_success))
            if n_success >= 2:
                logger.info(
                    "Stopping sampling on iteration {}/{} as n_effective>{}"
                    .format(ii, iterations, self.internal_kwargs["n_effective"]))
                self.nsteps = ii
                break
        self.checkpoint()

        #self.calculate_autocorrelation(self.sampler.chain.reshape((-1, self.ndim)))
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
        self.result.samples = self.sampler.chain[0, :, self.nburn:, :].reshape(
            (-1, self.ndim))
        self.result.walkers = self.sampler.chain[0, :, :, :]

        n_samples = self.nwalkers * self.nburn
        self.result.log_likelihood_evaluations = self.stored_loglike[n_samples:]
        self.result.log_prior_evaluations = self.stored_logprior[n_samples:]
        self.result.betas = self.sampler.betas
        self.result.log_evidence, self.result.log_evidence_err =\
            self.sampler.log_evidence_estimate(
                self.sampler.loglikelihood, self.nburn / self.nsteps)

        return self.result

from __future__ import absolute_import

import os
import signal
import time

import numpy as np

from .base_sampler import NestedSampler
from ..utils import logger


class PyPolyChord(NestedSampler):

    """
    Bilby wrapper of PyPolyChord
    https://arxiv.org/abs/1506.00171

    PolyChordLite is available at:
    https://github.com/PolyChord/PolyChordLite

    Follow the installation instructions at their github page.

    Keyword arguments will be passed into `pypolychord.run_polychord` into the `settings`
    argument. See the PolyChord documentation for what all of those mean.

    To see what the keyword arguments are for, see the docstring of PyPolyChordSettings
    """

    default_kwargs = dict(use_polychord_defaults=False, nlive=None, num_repeats=None,
                          nprior=-1, do_clustering=True, feedback=1, precision_criterion=0.001,
                          logzero=-1e30, max_ndead=-1, boost_posterior=0.0, posteriors=True,
                          equals=True, cluster_posteriors=True, write_resume=True,
                          write_paramnames=False, read_resume=True, write_stats=True,
                          write_live=True, write_dead=True, write_prior=True,
                          compression_factor=np.exp(-1), base_dir='outdir',
                          file_root='polychord', seed=-1, grade_dims=None, grade_frac=None, nlives={})

    def __init__(
            self, likelihood, priors, outdir='outdir', label='label',
            use_ratio=False, plot=False, skip_import_verification=False,
            injection_parameters=None, meta_data=None, result_class=None,
            likelihood_benchmark=False, soft_init=False, exit_code=130,
            **kwargs):

        super().__init__(
            likelihood, priors, outdir=outdir, label=label,
            use_ratio=use_ratio, plot=plot, skip_import_verification=skip_import_verification,
            injection_parameters=injection_parameters, meta_data=meta_data, result_class=result_class,
            likelihood_benchmark=likelihood_benchmark, soft_init=soft_init, exit_code=exit_code,
            **kwargs)

        try:
            signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
            signal.signal(signal.SIGINT, self.write_current_state_and_exit)
            signal.signal(signal.SIGALRM, self.write_current_state_and_exit)
        except AttributeError:
            logger.debug(
                "Setting signal attributes unavailable on this system. "
                "This is likely the case if you are running on a Windows machine"
                " and is no further concern.")

    def run_sampler(self):
        import pypolychord
        from pypolychord.settings import PolyChordSettings
        if self.kwargs['use_polychord_defaults']:
            settings = PolyChordSettings(nDims=self.ndim, nDerived=self.ndim,
                                         base_dir=self._sample_file_directory,
                                         file_root=self.label)
        else:
            self._setup_dynamic_defaults()
            pc_kwargs = self.kwargs.copy()
            pc_kwargs['base_dir'] = self._sample_file_directory
            pc_kwargs['file_root'] = self.label
            pc_kwargs.pop('use_polychord_defaults')
            settings = PolyChordSettings(nDims=self.ndim, nDerived=self.ndim, **pc_kwargs)
        self._verify_kwargs_against_default_kwargs()
        out = pypolychord.run_polychord(loglikelihood=self.log_likelihood, nDims=self.ndim,
                                        nDerived=self.ndim, settings=settings, prior=self.prior_transform)
        self.result.log_evidence = out.logZ
        self.result.log_evidence_err = out.logZerr
        log_likelihoods, physical_parameters = self._read_sample_file()
        self.result.log_likelihood_evaluations = log_likelihoods
        self.result.samples = physical_parameters
        self.calc_likelihood_count()
        return self.result

    def _setup_dynamic_defaults(self):
        """ Sets up some interdependent default argument if none are given by the user """
        if not self.kwargs['grade_dims']:
            self.kwargs['grade_dims'] = [self.ndim]
        if not self.kwargs['grade_frac']:
            self.kwargs['grade_frac'] = [1.0] * len(self.kwargs['grade_dims'])
        if not self.kwargs['nlive']:
            self.kwargs['nlive'] = self.ndim * 25
        if not self.kwargs['num_repeats']:
            self.kwargs['num_repeats'] = self.ndim * 5

    def _translate_kwargs(self, kwargs):
        if 'nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)

    def log_likelihood(self, theta):
        """ Overrides the log_likelihood so that PolyChord understands it """
        return super(PyPolyChord, self).log_likelihood(theta), theta

    def _read_sample_file(self):
        """
        This method reads out the _equal_weights.txt file that polychord produces.
        The first column is omitted since it is just composed of 1s, i.e. the equal weights.
        The second column are the log likelihoods, the remaining columns are the physical parameters.

        Returns
        -------
        array_like, array_like: The log_likelihoods and the associated parameters

        """
        sample_file = self._sample_file_directory + '/' + self.label + '_equal_weights.txt'
        samples = np.loadtxt(sample_file)
        log_likelihoods = -0.5 * samples[:, 1]
        physical_parameters = samples[:, -self.ndim:]
        return log_likelihoods, physical_parameters

    @property
    def _sample_file_directory(self):
        return self.outdir + '/chains'

    def write_current_state_and_exit(self, signum=None):
        if signum == 14:
            logger.info(
                "Run interrupted by alarm signal {}: checkpoint and exit on {}"
                .format(signum, self.exit_code))
        else:
            logger.info(
                "Run interrupted by signal {}: checkpoint and exit on {}"
                .format(signum, self.exit_code))
        self.write_current_state()
        os._exit(self.exit_code)

    def write_current_state(self):
        """
        Do nothing as the files get automatically written by Polychord.
        We wait 30 seconds to avoid corrupting files as they are being written.
        """
        time.sleep(30)

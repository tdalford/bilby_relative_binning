from __future__ import absolute_import

import os
import numpy as np
from pandas import DataFrame

from .base_sampler import NestedSampler
from .proposal import JumpProposal, JumpProposalCycle
from ..utils import logger, check_directory_exists_and_if_not_mkdir, infer_parameters_from_function

from .proposal import cpnest_proposal_factory, cpnest_proposal_cycle_factory


class Cpnest(NestedSampler):
    """ bilby wrapper of cpnest (https://github.com/johnveitch/cpnest)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `cpnest.CPNest`, see documentation
    for that class for further help. Under Other Parameters, we list commonly
    used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    nlive: int
        The number of live points, note this can also equivalently be given as
        one of [npoints, nlives, n_live_points]
    seed: int (1234)
        Initialised random seed
    nthreads: int, (1)
        Number of threads to use
    maxmcmc: int (1000)
        The maximum number of MCMC steps to take
    verbose: Bool (True)
        If true, print information information about the convergence during
    resume: Bool (False)
        Whether or not to resume from a previous run
    output: str
        Where to write the CPNest, by default this is
        {self.outdir}/cpnest_{self.label}/

    """
    default_kwargs = dict(verbose=1, nthreads=1, nlive=500, maxmcmc=1000,
                          seed=None, poolsize=100, nhamiltonian=0, resume=False,
                          output=None, proposals=None)

    def _translate_kwargs(self, kwargs):
        if 'nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)
        if 'seed' not in kwargs:
            logger.warning('No seed provided, cpnest will use 1234.')

    def run_sampler(self):
        from cpnest import model as cpmodel, CPNest
        from cpnest.cpnest import RunManager
        from cpnest.sampler import MetropolisHastingsSampler, HamiltonianMonteCarloSampler
        from cpnest.proposal import DefaultProposalCycle, HamiltonianProposalCycle
        from cpnest.NestedSampling import NestedSampler
        import multiprocessing as mp

        class GWCPNest(CPNest):

            def __init__(self,
                         usermodel,
                         nlive=100,
                         poolsize=100,
                         output='./',
                         verbose=0,
                         seed=None,
                         maxmcmc=100,
                         nthreads=None,
                         nhamiltonian=0,
                         resume=False,
                         proposals=None):

                if proposals is None:
                    proposals = dict(mhs=DefaultProposalCycle,
                                     hmc=HamiltonianProposalCycle)

                # super(GWCPNest, self).__init__(usermodel=usermodel,
                #                                nlive=nlive,
                #                                poolsize=poolsize,
                #                                output=output,
                #                                verbose=verbose,
                #                                seed=seed,
                #                                maxmcmc=maxmcmc,
                #                                nthreads=nthreads,
                #                                nhamiltonian=nhamiltonian,
                #                                resume=resume)
                if nthreads is None:
                    self.nthreads = mp.cpu_count()
                else:
                    self.nthreads = nthreads
                print('Running with {0} parallel threads'.format(self.nthreads))
                self.user = usermodel
                self.nlive = nlive
                self.verbose = verbose
                self.output = output
                self.poolsize = poolsize
                self.posterior_samples = None
                self.resume = resume

                if seed is None:
                    self.seed = 1234
                else:
                    self.seed = seed

                self.manager = RunManager(nthreads=self.nthreads)
                self.manager.start()

                self.process_pool = []

                resume_file = os.path.join(output, "nested_sampler_resume.pkl")
                if not os.path.exists(resume_file) or resume is False:
                    self.NS = NestedSampler(self.user,
                                            nlive=nlive,
                                            output=output,
                                            verbose=verbose,
                                            seed=self.seed,
                                            prior_sampling=False,
                                            manager=self.manager)
                else:
                    self.NS = NestedSampler.resume(resume_file, self.manager, self.user)

                for i in range(self.nthreads - nhamiltonian):
                    resume_file = os.path.join(output, "sampler_{0:d}.pkl".format(i))
                    if not os.path.exists(resume_file) or resume is False:
                        sampler = MetropolisHastingsSampler(self.user,
                                                            maxmcmc,
                                                            verbose=verbose,
                                                            output=output,
                                                            poolsize=poolsize,
                                                            seed=self.seed + i,
                                                            proposal=proposals['mhs'](),
                                                            resume_file=resume_file,
                                                            manager=self.manager
                                                            )
                    else:
                        sampler = MetropolisHastingsSampler.resume(resume_file,
                                                                   self.manager,
                                                                   self.user)

                    p = mp.Process(target=sampler.produce_sample)
                    self.process_pool.append(p)

                for i in range(self.nthreads - nhamiltonian, self.nthreads):
                    resume_file = os.path.join(output, "sampler_{0:d}.pkl".format(i))
                    if not os.path.exists(resume_file) or resume is False:
                        sampler = HamiltonianMonteCarloSampler(self.user,
                                                               maxmcmc,
                                                               verbose=verbose,
                                                               output=output,
                                                               poolsize=poolsize,
                                                               seed=self.seed + i,
                                                               proposal=proposals['hmc'](self.user),
                                                               resume_file=resume_file,
                                                               manager=self.manager
                                                               )
                    else:
                        sampler = HamiltonianMonteCarloSampler.resume(resume_file,
                                                                      self.manager,
                                                                      self.user)
                    p = mp.Process(target=sampler.produce_sample)
                    self.process_pool.append(p)

        class Model(cpmodel.Model):
            """ A wrapper class to pass our log_likelihood into cpnest """

            def __init__(self, names, bounds):
                self.names = names
                self.bounds = bounds
                self._check_bounds()

            @staticmethod
            def log_likelihood(x, **kwargs):
                theta = [x[n] for n in self.search_parameter_keys]
                return self.log_likelihood(theta)

            @staticmethod
            def log_prior(x, **kwargs):
                theta = [x[n] for n in self.search_parameter_keys]
                return self.log_prior(theta)

            def _check_bounds(self):
                for bound in self.bounds:
                    if not all(np.isfinite(bound)):
                        raise ValueError(
                            'CPNest requires priors to have finite bounds.')

        bounds = [[self.priors[key].minimum, self.priors[key].maximum]
                  for key in self.search_parameter_keys]
        self._resolve_proposal_functions()

        model = Model(self.search_parameter_keys, bounds)
        out = GWCPNest(model, **self.kwargs)
        out.run()

        if self.plot:
            out.plot()

        self.result.posterior = DataFrame(out.posterior_samples)
        self.result.posterior.rename(columns=dict(
            logL='log_likelihood', logPrior='log_prior'), inplace=True)
        self.result.log_evidence = out.NS.state.logZ
        self.result.log_evidence_err = np.nan
        return self.result

    def _verify_kwargs_against_default_kwargs(self):
        """
        Set the directory where the output will be written.
        """
        if not self.kwargs['output']:
            self.kwargs['output'] = \
                '{}/cpnest_{}/'.format(self.outdir, self.label)
        if self.kwargs['output'].endswith('/') is False:
            self.kwargs['output'] = '{}/'.format(self.kwargs['output'])
        check_directory_exists_and_if_not_mkdir(self.kwargs['output'])
        NestedSampler._verify_kwargs_against_default_kwargs(self)

    def _resolve_proposal_functions(self):
        from cpnest.proposal import Proposal
        if 'proposals' in self.kwargs:
            if self.kwargs['proposals'] is None:
                return
            for key, proposal in self.kwargs['proposals'].items():
                if isinstance(proposal, JumpProposal):
                    self.kwargs['proposals'][key] = cpnest_proposal_factory(proposal)
                elif isinstance(proposal, JumpProposalCycle):
                    self.kwargs['proposals'][key] = cpnest_proposal_cycle_factory(proposal)
                elif isinstance(proposal, Proposal):
                    pass
                else:
                    raise TypeError("Unknown proposal type")

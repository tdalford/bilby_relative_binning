import random
from random import sample as random_sample
from random import choice, uniform, gauss, randrange
from functools import reduce
import numpy as np

from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal

import cpnest.proposal
from emcee.moves import MHMove

# MIT License
#
# Copyright (c) 2015-2016 Walter Del Pozzo, John Veitch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class JumpProposal(object):

    def __init__(self, proposal_function):
        self.proposal_function = proposal_function

    def __call__(self, coordinates, **kwargs):
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return self.proposal_function(coordinates=coordinates, **kwargs)


class JumpProposalCycle(object):

    def __init__(self, proposal_functions, weights, cycle_length):
        assert len(proposal_functions) == len(weights)
        self.proposal_functions = proposal_functions
        self._weights = weights
        self.cycle_length = cycle_length
        self.index = 0
        self.cycle = np.random.choice(self.proposal_functions, size=self.cycle_length,
                                      p=self.weights, replace=True)

    def __call__(self, coordinates, **kwargs):
        proposal = self.cycle[self.index]
        new = proposal(coordinates, **kwargs)
        self.index = (self.index + 1) % self.cycle_length
        return new

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = np.array(weights)/np.sum(np.array(weights))


def cpnest_proposal_factory(jump_proposal, **kwargs):
    class CPNestEnsembleProposal(cpnest.proposal.EnsembleProposal):

        def __init__(self):
            self.log_J = kwargs.get('log_j', 0)
            if 'log_j' in kwargs:
                del kwargs['log_j']
            self.kwargs = kwargs

        def get_sample(self, old):
            return jump_proposal(sample=old, coordinates=self.ensemble, **kwargs)

        def set_ensemble(self, ensemble):
            self.ensemble = ensemble

    return CPNestEnsembleProposal


def cpnest_proposal_cycle_factory(jump_proposals, weights, **kwargs):
    cpnest_jump_proposals = [cpnest_proposal_factory(jp) for jp in jump_proposals]

    class CPNestProposalCycle(cpnest.proposal.ProposalCycle):

        def __init__(self):
            super().__init__(proposals=cpnest_jump_proposals,
                             weights=weights, **kwargs)

    return CPNestProposalCycle


def emcee_proposal_factory(jump_proposal, **kwargs):

    def jump_proposal_wrapper(random_number_generator, coordinates):
        return jump_proposal(coordinates)

    MHMove(proposal_function=jump_proposal_wrapper)


def ptmcmc_proposal_factory(jump_proposal, weight, proposal_name=None):
    return ptmcmc_proposal_cycle_factory([jump_proposal], [weight], [proposal_name])


def ptmcmc_proposal_cycle_factory(jump_proposals, weights, proposal_names=None):
    if proposal_names is None:
        custom = {jump_proposals[i].__name__: [jump_proposals[i], weights[i]] for i in range(jump_proposals)}
    else:
        custom = {proposal_names[i]: [jump_proposals[i], weights[i]] for i in range(jump_proposals)}
    return custom


class UniformJump(object):

    def __init__(self, pmin, pmax):
        """Draw random parameters from pmin, pmax"""
        self.min = pmin
        self.max = pmax

    def __call__(self, sample):
        """
        Function prototype must read in parameter vector x,
        sampler iteration number it, and inverse temperature beta
        """
        q = np.random.uniform(self.min, self.max, len(x))
        return q


class NormJump(object):
    def __init__(self, step_size):
        """Draw random parameters from pmin, pmax"""
        self.step_size = step_size

    def __call__(self, sample):
        q = np.random.multivariate_normal(sample, self.step_size * np.eye(len(sample)), 1)
        return q[0]


class EnsembleWalk(object):

    def __init__(self, random_number_generator=None, npoints=3):
        if random_number_generator is None:
            self.random_number_generator = random.random
        self.npoints = npoints

    def __call__(self, sample, coordinates, **kwargs):
        subset = random_sample(coordinates, self.npoints)
        center_of_mass = reduce(type(sample).__add__, subset) / float(self.npoints)
        out = sample
        for x in subset:
            out += (x - center_of_mass) * self.random_number_generator()
        return out


class EnsembleWalkDegeneracy(EnsembleWalk):

    def __call__(self, sample, coordinates, **kwargs):
        subset = random_sample(coordinates, self.npoints)
        center_of_mass = reduce(type(sample).__add__, subset) / float(self.npoints)
        out = sample
        for x in subset:
            out['mass_1'] += (x['mass_1'] - center_of_mass['mass_1']) * self.random_number_generator()
            out['phase'] += (x['phase'] - center_of_mass['phase']) * self.random_number_generator()
            if self.random_number_generator() > 0.5:
                if out['phase'] > np.pi:
                    out['phase'] -= np.pi
                else:
                    out['phase'] += np.pi
        return out


class EnsembleStretch(object):

    def __call__(self, sample, coordinates, **kwargs):
        scale = 2.0
        a = choice(coordinates)
        x = uniform(-1, 1) * np.log(scale)
        Z = np.exp(x)
        out = a + (sample - a) * Z
        self.log_j = out.dimension * x
        return out


class DifferentialEvolution(object):

    def __call__(self, sample, coordinates, **kwargs):
        """
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
        a, b = random_sample(coordinates, 2)
        sigma = 1e-4  # scatter around difference vector by this factor
        out = sample + (b - a) * gauss(1.0, sigma)
        return out


class EnsembleEigenVector(object):

    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None
        self.covariance = None

    def update_eigenvectors(self, coordinates):
        """
        Recompute the eigenvectors and eigevalues
        of the covariance matrix of the ensemble
        """
        n = len(coordinates)
        dim = coordinates[0].dimension
        cov_array = np.zeros((dim, n))
        if dim == 1:
            name = coordinates[0].names[0]
            self.eigen_values = np.atleast_1d(np.var([coordinates[j][name] for j in range(n)]))
            self.covariance = self.eigen_values
            self.eigen_vectors = np.eye(1)
        else:
            for i, name in enumerate(coordinates[0].names):
                for j in range(n): cov_array[i, j] = coordinates[j][name]
            self.covariance = np.cov(cov_array)
            self.eigen_values, self.eigen_vectors = np.linalg.eigh(self.covariance)

    def __call__(self, sample, coordinates):
        self.update_eigenvectors(coordinates)
        out = sample
        i = randrange(sample.dimension)
        jumpsize = np.sqrt(np.fabs(self.eigen_values[i])) * gauss(0, 1)
        for k, n in enumerate(out.names):
            out[n] += jumpsize * self.eigen_vectors[k, i]
        return out


class HamiltonianProposal(EnsembleEigenVector):

    def __init__(self, model=None):
        super().__init__()
        self.mass_matrix = None
        self.inverse_mass_matrix = None
        self.momenta_distribution = None
        self.T = self.kinetic_energy
        self.V = model.potential
        self.normal = None

    def __call__(self, sample, coordinates):
        self.update_mass()
        self.update_normal_vector(coordinates)
        self.update_momenta_distribution()
        super(HamiltonianProposal, self).__call__(sample, coordinates)

    def update_normal_vector(self, coordinates):
        """
        update the constraint by approximating the
        loglikelihood hypersurface as a spline in
        each dimension.
        This is an approximation which
        improves as the algorithm proceeds
        """
        n = coordinates[0].dimension
        tracers_array = np.zeros((len(coordinates), n))
        for i, samp in enumerate(coordinates):
            tracers_array[i, :] = samp.values
        V_vals = np.atleast_1d([p.logL for p in coordinates])

        self.normal = []
        for i, x in enumerate(tracers_array.T):
            # sort the values
            idx = x.argsort()
            xs = x[idx]
            Vs = V_vals[idx]
            # remove potential duplicate entries
            xs, ids = np.unique(xs, return_index=True)
            Vs = Vs[ids]
            # pick only finite values
            idx = np.isfinite(Vs)
            Vs = Vs[idx]
            xs = xs[idx]
            # filter to within the 90% range of the Pvals
            Vl, Vh = np.percentile(Vs, [5, 95])
            (idx,) = np.where(np.logical_and(Vs > Vl, Vs < Vh))
            Vs = Vs[idx]
            xs = xs[idx]
            # Pick knots for this parameters: Choose 5 knots between
            # the 1st and 99th percentiles (heuristic tuning WDP)
            knots = np.percentile(xs, np.linspace(1, 99, 5))
            # Guesstimate the length scale for numerical derivatives
            dimwidth = knots[-1] - knots[0]
            delta = 0.1 * dimwidth / len(idx)
            # Apply a Savtzky-Golay filter to the likelihoods (low-pass filter)
            window_length = len(idx) // 2 + 1  # Window for Savtzky-Golay filter
            if window_length % 2 == 0: window_length += 1
            f = savgol_filter(Vs, window_length,
                              5,  # Order of polynominal filter
                              deriv=1,  # Take first derivative
                              delta=delta,  # delta for numerical deriv
                              mode='mirror'  # Reflective boundary conds.
                              )
            # construct a LSQ spline interpolant
            self.normal.append(LSQUnivariateSpline(xs, f, knots, ext=3, k=3))

    def unit_normal(self, q):
        """
        Returns the unit normal to the iso-Likelihood surface
        at x, obtained from the spline interpolation of the
        directional derivatives of the likelihood
        Parameters
        ----------
        q : :obj:`cpnest.parameter.LivePoint`
            position

        Returns
        ----------
        n: :obj:`numpy.ndarray` unit normal to the logLmin contour evaluated at q
        """
        v = np.array([self.normal[i](q[n]) for i, n in enumerate(q.names)])
        v[np.isnan(v)] = -1.0
        n = v / np.linalg.norm(v)
        return n

    def gradient(self, q):
        """
        return the gradient of the potential function as numpy ndarray
        Parameters
        ----------
        q : :obj:`cpnest.parameter.LivePoint`
            position

        Returns
        ----------
        dV: :obj:`numpy.ndarray` gradient evaluated at q
        """
        dV = self.dV(q)
        return dV.view(np.float64)

    def update_momenta_distribution(self):
        """
        update the momenta distribution using the
        mass matrix (precision matrix of the ensemble).
        """
        self.momenta_distribution = multivariate_normal(cov=self.mass_matrix)

    def update_mass(self):
        """
        Update the mass matrix (covariance matrix) and
        inverse mass matrix (precision matrix)
        from the ensemble, allowing for correlated momenta
        """
        self.mass_matrix = np.linalg.inv(np.atleast_2d(self.covariance))
        self.inverse_mass_matrix = np.atleast_2d(self.covariance)

    def kinetic_energy(self, p):
        """
        kinetic energy part for the Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum

        Returns
        ----------
        T: :float: kinetic energy
        """
        return 0.5 * np.dot(p, np.dot(self.inverse_mass_matrix, p))

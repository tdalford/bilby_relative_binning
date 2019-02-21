import random
from random import sample as random_sample
from random import choice, uniform, gauss, randrange
from functools import reduce
import numpy as np


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

    def __init__(self, proposal_functions, weights, cycle_length=100):
        self.proposal_functions = proposal_functions
        self.weights = weights
        self.cycle_length = cycle_length
        self._index = 0
        self.cycle = np.random.choice(self.proposal_functions, size=self.cycle_length,
                                      p=self.weights, replace=True)

    def __call__(self, coordinates, **kwargs):
        proposal = self.cycle[self.index]
        new = proposal(coordinates=coordinates, **kwargs)
        self._index = (self.index + 1) % self.cycle_length
        return new

    def __len__(self):
        return len(self.proposal_functions)

    @property
    def index(self):
        return self._index

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        assert len(weights) == len(self.proposal_functions)
        self._weights = np.array(weights)/np.sum(np.array(weights))


class UniformJump(object):

    def __init__(self, pmin, pmax):
        """Draw random parameters from pmin, pmax"""
        self.pmin = pmin
        self.pmax = pmax

    def __call__(self, sample):
        """
        Function prototype must read in parameter vector x,
        sampler iteration number it, and inverse temperature beta
        """
        return np.random.uniform(self.pmin, self.pmax, len(sample))


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

    def __call__(self, sample, coordinates, **kwargs):
        self.update_eigenvectors(coordinates)
        out = sample
        i = randrange(sample.dimension)
        jumpsize = np.sqrt(np.fabs(self.eigen_values[i])) * gauss(0, 1)
        for k, n in enumerate(out.names):
            out[n] += jumpsize * self.eigen_vectors[k, i]
        return out

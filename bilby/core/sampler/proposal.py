import random
from random import sample as random_sample
from random import choice, uniform, gauss, randrange
from functools import reduce
import numpy as np
from inspect import isclass


class JumpProposalWrapper(object):

    def __init__(self, proposal_function):
        """ A generic wrapper class for jump proposals

        Parameters
        ----------
        proposal_function: callable
        A callable object or function that returns the proposal
        """
        self.proposal_function = proposal_function

    def __call__(self, *args, **kwargs):
        """ A generic wrapper for the jump proposal function

        Parameters
        ----------
        args: Arguments that are going to be passed into the proposal function
        kwargs: Keyword arguments that are going to be passed into the proposal function

        Returns
        -------

        """
        return self.proposal_function(*args, **kwargs)


class JumpProposalCycleWrapper(object):

    def __init__(self, proposal_functions, weights, cycle_length=100):
        """ A generic wrapper class for proposal cycles

        Parameters
        ----------
        proposal_functions: list
        A list of callable proposal functions/objects
        weights: list

        cycle_length
        """
        self.proposal_functions = proposal_functions
        self.weights = weights
        self.cycle_length = cycle_length
        self._index = 0
        self._cycle = np.random.choice(self.proposal_functions, size=self.cycle_length,
                                       p=self.weights, replace=True)

    def __call__(self, coordinates, **kwargs):
        proposal = self._cycle[self.index]
        self._index = (self.index + 1) % self.cycle_length
        return proposal(coordinates=coordinates, **kwargs)

    def __len__(self):
        return len(self.proposal_functions)

    @property
    def proposal_functions(self):
        return self._proposal_functions

    @proposal_functions.setter
    def proposal_functions(self, proposal_functions):
        for i, proposal in enumerate(proposal_functions):
            if isclass(proposal):
                proposal_functions[i] = proposal()
        self._proposal_functions = proposal_functions

    @property
    def index(self):
        return self._index

    @property
    def weights(self):
        return np.array(self._weights) / np.sum(np.array(self._weights))

    @weights.setter
    def weights(self, weights):
        assert len(weights) == len(self.proposal_functions)
        self._weights = weights

    @property
    def unnormalised_weights(self):
        return self._weights


class UniformJump(object):

    def __init__(self, pmin=0, pmax=1):
        self.pmin = pmin
        self.pmax = pmax

    def __call__(self, sample, *args, **kwargs):
        return np.random.uniform(self.pmin, self.pmax, len(sample))


class NormJump(object):
    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, sample, *args, **kwargs):
        q = np.random.multivariate_normal(sample, self.step_size * np.eye(len(sample)), 1)
        return q[0]


class EnsembleWalk(object):

    def __init__(self, random_number_generator=random.random, npoints=3, **random_number_generator_args):
        self.random_number_generator = random_number_generator
        self.npoints = npoints
        self.random_number_generator_args = random_number_generator_args

    def __call__(self, sample, coordinates, *args, **kwargs):
        subset = random_sample(coordinates, self.npoints)
        center_of_mass = reduce(type(sample).__add__, subset) / float(self.npoints)
        out = sample
        for x in subset:
            out += (x - center_of_mass) * self.random_number_generator(**self.random_number_generator_args)
        return out


class GWEnsembleWalkPrototype(EnsembleWalk):

    def _get_sky_keys(self, sample):
        return self._get_keys(['ra', 'dec'], sample)

    def _get_cyclical_keys(self, sample):
        return self._get_keys(['phase', 'psi'], sample)

    @staticmethod
    def _get_keys(keys, sample):
        return list(set(keys) & set(sample.keys()))

    def __call__(self, sample, coordinates, **kwargs):
        subset = random_sample(coordinates, self.npoints)
        center_of_mass = reduce(type(sample).__add__, subset) / float(self.npoints)
        out = sample
        for x in subset:
            out += (x - center_of_mass) * self.random_number_generator()

            self._move_phase_psi(out)
            self._move_ra_dec(out)

        return out

    def _move_phase_psi(self, out):
        for key in self._get_cyclical_keys(out):
            if self.random_number_generator() > 0.5:
                self._reflect_by_pi(key, out)

    @staticmethod
    def _reflect_by_pi(key, out):
        if out[key] > np.pi:
            out[key] -= np.pi
        else:
            out[key] += np.pi

    def _move_ra_dec(self, out):
        keys = self._get_sky_keys(out)
        if self.random_number_generator() > 0.5:
            if 'ra' in keys:
                self._reflect_by_pi('ra', out)
            if 'dec' in keys:
                out['dec'] = -out['dec']


class EnsembleStretch(object):

    def __init__(self, scale=2.0):
        self.scale = scale

    def __call__(self, sample, coordinates, **kwargs):
        second_sample = choice(coordinates)
        random_number = uniform(-1, 1) * np.log(self.scale)
        out = second_sample + (sample - second_sample) * np.exp(random_number)
        self.log_j = out.dimension * random_number
        return out


class DifferentialEvolution(object):

    def __call__(self, sample, coordinates, **kwargs):
        a, b = random_sample(coordinates, 2)
        sigma = 1e-4
        return sample + (b - a) * gauss(1.0, sigma)


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

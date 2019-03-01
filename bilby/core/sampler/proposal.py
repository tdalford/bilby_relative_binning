import random
from functools import reduce
import numpy as np
from inspect import isclass


class JumpProposal(object):

    def __init__(self, prior=None, log_likelihood=None):
        """ A generic wrapper class for jump proposals

        Parameters
        ----------
        proposal_function: callable
        A callable object or function that returns the proposal
        """
        self.prior = prior
        self.log_likelihood = log_likelihood

    def __call__(self, *args, **kwargs):
        """ A generic wrapper for the jump proposal function

        Parameters
        ----------
        args: Arguments that are going to be passed into the proposal function
        kwargs: Keyword arguments that are going to be passed into the proposal function

        Returns
        -------

        """
        pass


class JumpProposalCycleWrapper(object):

    def __init__(self, proposal_functions, weights, cycle_length=100):
        """ A generic wrapper class for proposal cycles

        Parameters
        ----------
        proposal_functions: list
        A list of callable proposal functions/objects
        weights: list
        A list of integer weights for the respective proposal functions
        cycle_length: int, optional
        Length of the proposal cycle
        """
        self.proposal_functions = proposal_functions
        self.weights = weights
        self.cycle_length = cycle_length
        self._index = 0
        self._cycle = np.random.choice(self.proposal_functions, size=self.cycle_length,
                                       p=self.weights, replace=True)

    def __call__(self, **kwargs):
        proposal = self._cycle[self.index]
        self._index = (self.index + 1) % self.cycle_length
        return proposal(**kwargs)

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
        """

        Returns
        -------
        Normalised proposal weights

        """
        return np.array(self._weights) / np.sum(np.array(self._weights))

    @weights.setter
    def weights(self, weights):
        assert len(weights) == len(self.proposal_functions)
        self._weights = weights

    @property
    def unnormalised_weights(self):
        return self._weights


class UniformJump(JumpProposal):

    def __init__(self, pmin=0, pmax=1, prior=None, log_likelihood=None):
        """
        A primitive uniform jump
        Parameters
        ----------
        pmin: float, optional
        The minimum boundary of the uniform jump
        pmax: float, optional
        The maximum boundary of the uniform jump
        """
        super().__init__(prior, log_likelihood)
        self.pmin = pmin
        self.pmax = pmax
        self.prior = prior
        self.log_likelihood = log_likelihood

    def __call__(self, sample, *args, **kwargs):
        new = np.random.uniform(self.pmin, self.pmax, len(sample))
        self.proposal_probability = 0
        return new


class NormJump(JumpProposal):
    def __init__(self, step_size, prior=None, log_likelihood=None):
        """
        A normal distributed step centered around the old sample

        Parameters
        ----------
        step_size: float
        The scalable step size
        """
        super().__init__(prior, log_likelihood)
        self.step_size = step_size

    def __call__(self, sample, *args, **kwargs):
        q = np.random.multivariate_normal(sample, self.step_size * np.eye(len(sample)), 1)
        self.proposal_probability = 0
        return q[0]


# class ArbitraryJump(object):
#     def __init__(self, random_number_generator, **random_number_generator_kwargs):
#         """
#
#         Parameters
#         ----------
#         random_number_generator: func
#         A random number generator that needs to wrapped so that the first element is the old sample
#         random_number_generator_kwargs:
#         Additional keyword arguments that go into the random number generator
#         """
#         self.random_number_generator = random_number_generator
#         self.random_number_generator_kwargs = random_number_generator_kwargs
#
#     def __call__(self, sample, *args, **kwargs):
#         return self.random_number_generator(sample, **self.random_number_generator_kwargs)
#

class EnsembleWalk(JumpProposal):

    def __init__(self, random_number_generator=random.random, npoints=3, prior=None, log_likelihood=None,
                 **random_number_generator_args):
        """
        An ensemble walk
        Parameters
        ----------
        random_number_generator: func, optional
        A random number generator. Default is random.random
        npoints: int, optional
        Number of points in the ensemble to average over. Default is 3.
        random_number_generator_args:
        Additional keyword arguments for the random number generator
        """
        super().__init__(prior, log_likelihood)
        self.random_number_generator = random_number_generator
        self.npoints = npoints
        self.random_number_generator_args = random_number_generator_args

    def __call__(self, sample, coordinates, *args, **kwargs):
        subset = random.sample(coordinates, self.npoints)
        center_of_mass = reduce(type(sample).__add__, subset) / float(self.npoints)
        out = sample
        for x in subset:
            out += (x - center_of_mass) * self.random_number_generator(**self.random_number_generator_args)
        return out


class GWEnsembleWalkPrototype(EnsembleWalk):
    """
    A prototype implementation to demonstrate how a random ensemble walk could be modified for
    gravitational wave signals.
    """

    def __call__(self, sample, coordinates, prior=None, log_likelihood=None, **kwargs):
        subset = random.sample(coordinates, self.npoints)
        center_of_mass = reduce(type(sample).__add__, subset) / float(self.npoints)
        out = sample
        for x in subset:
            out += (x - center_of_mass) * self.random_number_generator()

            out = self._move_degenerate_keys(out)
            out = self._move_ra_dec(out)
            out = self._move_periodic_keys(out)

        return out

    def _move_degenerate_keys(self, out):
        for key in self._get_degenerate_keys(out):
            if self.random_number_generator() > 0.5:
                out = self._reflect_by_pi(key, out)
        return out

    def _move_periodic_keys(self, out):
        for key in self._get_periodic_keys(out):
            # periodic move, maybe we should use while loops instead???
            if out[key] > self.prior[key].maximum:
                out[key] = self.prior[key].minimum + out[key] - self.prior[key].maximum
            elif out[key] < self.prior[key].minimum:
                out[key] = self.prior[key].maximum - out[key] + self.prior[key].minimum
        return out

    def _move_ra_dec(self, out):
        keys = self._get_sky_keys(out)
        if self.random_number_generator() > 0.5:
            if 'ra' in keys:
                self._reflect_by_pi('ra', out)
            if 'dec' in keys:
                out['dec'] = -out['dec']
        return out

    @staticmethod
    def _reflect_by_pi(key, out):
        if out[key] > np.pi:
            out[key] -= np.pi
        else:
            out[key] += np.pi
        return out

    def _get_sky_keys(self, sample):
        return self._get_keys(['ra', 'dec'], sample)

    def _get_periodic_keys(self, sample):
        return self._get_keys(['ra', 'phase', 'psi'], sample)

    def _get_degenerate_keys(self, sample):
        return self._get_keys(['phase', 'psi'], sample)

    @staticmethod
    def _get_keys(keys, sample):
        return list(set(keys) & set(sample.keys()))


class EnsembleStretch(JumpProposal):

    def __init__(self, scale=2.0, prior=None, log_likelihood=None):
        """
        Stretch move. Calculates the log Jacobian which can be used in cpnest to bias future moves.

        Parameters
        ----------
        scale: float, optional
        Stretching scale. Default is 2.0.
        """
        super().__init__(prior, log_likelihood)
        self.scale = scale

    def __call__(self, sample, coordinates, **kwargs):
        second_sample = random.choice(coordinates)
        step = random.uniform(-1, 1) * np.log(self.scale)
        out = second_sample + (sample - second_sample) * np.exp(step)
        self.log_j = out.dimension * step
        return out


class DifferentialEvolution(JumpProposal):

    def __init__(self, sigma=1e-4, mu=1.0, prior=None, log_likelihood=None):
        """
        Differential evolution step. Takes two elements from the existing coordinates and differentially evolves the
        old sample based on them using some Gaussian randomisation in the step.

        Parameters
        ----------
        sigma: float, optional
        Random spread in the evolution step. Default is 1e-4
        mu: float, optional
        Scale of the randomization. Default is 1.0
        """
        super().__init__(prior, log_likelihood)
        self.sigma = sigma
        self.mu = mu

    def __call__(self, sample, coordinates, **kwargs):
        a, b = random.sample(coordinates, 2)
        return sample + (b - a) * random.gauss(self.mu, self.sigma)


class EnsembleEigenVector(JumpProposal):

    def __init__(self, prior=None, log_likelihood=None):
        """
        Ensemble step based on the ensemble eigen vectors.
        """
        super().__init__(prior, log_likelihood)
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
                for j in range(n):
                    cov_array[i, j] = coordinates[j][name]
            self.covariance = np.cov(cov_array)
            self.eigen_values, self.eigen_vectors = np.linalg.eigh(self.covariance)

    def __call__(self, sample, coordinates, **kwargs):
        self.update_eigenvectors(coordinates)
        out = sample
        i = random.randrange(sample.dimension)
        jumpsize = np.sqrt(np.fabs(self.eigen_values[i])) * random.gauss(0, 1)
        for k, n in enumerate(out.names):
            out[n] += jumpsize * self.eigen_vectors[k, i]
        return out

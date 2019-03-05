import random
from functools import reduce
import numpy as np
import copy
from inspect import isclass

from bilby.core.prior import Uniform


class JumpProposal(object):

    def __init__(self, priors=None):
        """ A generic class for jump proposals

        Parameters
        ----------
        priors: bilby.core.prior.PriorDict
            Dictionary of priors used in this sampling run

        Attributes
        ----------
        log_j: float
            Log Jacobian of the proposal. Characterises whether or not detailed balance
            is preserved. If not, log_j needs to be adjusted accordingly.
        """
        self.priors = priors
        self.log_j = 0.0

    def __call__(self, *args, **kwargs):
        """ A generic wrapper for the jump proposal function

        Parameters
        ----------
        args: Arguments that are going to be passed into the proposal function
        kwargs: Keyword arguments that are going to be passed into the proposal function

        Returns
        -------
        dict: A dictionary with the new samples. Boundary conditions are applied.

        """
        return self._apply_boundaries(copy.copy(args[0]))

    def _move_reflecting_keys(self, out):
        keys = [key for key in self.priors.keys() if self.priors[key].boundary == 'reflecting']
        for key in keys:
            if out[key] > self.priors[key].maximum:
                out[key] = 2 * self.priors[key].maximum - out[key]
            elif out[key] < self.priors[key].minimum:
                out[key] = 2 * self.priors[key].minimum - out[key]
        return out

    def _move_periodic_keys(self, out):
        keys = [key for key in self.priors.keys() if self.priors[key].boundary == 'periodic']
        for key in keys:
            if out[key] > self.priors[key].maximum:
                out[key] = self.priors[key].minimum + out[key] - self.priors[key].maximum
            elif out[key] < self.priors[key].minimum:
                out[key] = self.priors[key].maximum + out[key] - self.priors[key].minimum
        return out

    def _apply_boundaries(self, sample):
        out = copy.copy(sample)
        out = self._move_periodic_keys(out)
        out = self._move_reflecting_keys(out)
        return out


class JumpProposalCycle(object):

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
        self._cycle = np.array([])
        self.update_cycle()

    def __call__(self, **kwargs):
        proposal = self._cycle[self.index]
        self._index = (self.index + 1) % self.cycle_length
        return proposal(**kwargs)

    def __len__(self):
        return len(self.proposal_functions)

    def update_cycle(self):
        self._cycle = np.random.choice(self.proposal_functions, size=self.cycle_length,
                                       p=self.weights, replace=True)

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


class NormJump(JumpProposal):
    def __init__(self, step_size, priors=None):
        """
        A normal distributed step centered around the old sample

        Parameters
        ----------
        step_size: float
            The scalable step size
        priors:
            See superclass
        """
        super(NormJump, self).__init__(priors)
        self.step_size = step_size

    def __call__(self, sample, *args, **kwargs):
        out = copy.copy(sample)
        for key in out.keys():
            out[key] = np.random.normal(sample[key], self.step_size)
        return super(NormJump, self).__call__(out)


class EnsembleWalk(JumpProposal):

    def __init__(self, random_number_generator=random.random, n_points=3, priors=None,
                 **random_number_generator_args):
        """
        An ensemble walk
        Parameters
        ----------
        random_number_generator: func, optional
            A random number generator. Default is random.random
        n_points: int, optional
            Number of points in the ensemble to average over. Default is 3.
        priors:
            See superclass
        random_number_generator_args:
            Additional keyword arguments for the random number generator
        """
        super(EnsembleWalk, self).__init__(priors)
        self.random_number_generator = random_number_generator
        self.n_points = n_points
        self.random_number_generator_args = random_number_generator_args

    def __call__(self, sample, coordinates, *args, **kwargs):
        out = copy.copy(sample)
        subset = random.sample(coordinates, self.n_points)
        center_of_mass = self.get_center_of_mass(subset)
        for x in subset:
            random_number = self.random_number_generator(**self.random_number_generator_args)
            for key in out:
                out[key] += (x[key] - center_of_mass[key]) * random_number
        return super(EnsembleWalk, self).__call__(out)

    @staticmethod
    def get_center_of_mass(subset):
        return {key: np.mean([c[key] for c in subset]) for key in subset[0].keys()}


class EnsembleStretch(JumpProposal):

    def __init__(self, scale=2.0, priors=None):
        """
        Stretch move. Calculates the log Jacobian which can be used in cpnest to bias future moves.

        Parameters
        ----------
        scale: float, optional
            Stretching scale. Default is 2.0.
        """
        super(EnsembleStretch, self).__init__(priors)
        self.scale = scale

    def __call__(self, sample, coordinates, **kwargs):
        out = copy.copy(sample)
        second_sample = random.choice(coordinates)
        step = random.uniform(-1, 1) * np.log(self.scale)
        for key in out.keys():
            out[key] = second_sample[key] + (sample[key] - second_sample[key]) * np.exp(step)
        self.log_j = out.dimension * step
        return super(EnsembleStretch, self).__call__(out)


class DifferentialEvolution(JumpProposal):

    def __init__(self, sigma=1e-4, mu=1.0, priors=None):
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
        super(DifferentialEvolution, self).__init__(priors)
        self.sigma = sigma
        self.mu = mu

    def __call__(self, sample, coordinates, **kwargs):
        a, b = random.sample(coordinates, 2)
        out = sample + (b - a) * random.gauss(self.mu, self.sigma)
        return super(DifferentialEvolution, self).__call__(out)


class EnsembleEigenVector(JumpProposal):

    def __init__(self, priors=None):
        """
        Ensemble step based on the ensemble eigenvectors.

        Parameters
        ----------
        priors:
            See superclass
        """
        super(EnsembleEigenVector, self).__init__(priors)
        self.eigen_values = None
        self.eigen_vectors = None
        self.covariance = None

    def update_eigenvectors(self, coordinates):
        if coordinates is None:
            return
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
        i = random.randrange(len(sample))
        jump_size = np.sqrt(np.fabs(self.eigen_values[i])) * random.gauss(0, 1)
        for k, n in enumerate(out.names):
            out[n] += jump_size * self.eigen_vectors[k, i]
        return super(EnsembleEigenVector, self).__call__(out)


class SkyLocationWanderJump(JumpProposal):
    """
    Jump proposal for wandering over the sky location. Does a Gaussian step in
    RA and DEC depending on the temperature.
    """

    def __call__(self, sample, **kwargs):
        temperature = 1 / kwargs.get('inverse_temperature', 1.0)
        out = copy.copy(sample)
        sigma = np.sqrt(temperature) / 2 / np.pi
        out['ra'] += random.gauss(0, sigma)
        out['dec'] += random.gauss(0, sigma)
        return super(SkyLocationWanderJump, self).__call__(out)


class CorrelatedPolarisationPhaseJump(JumpProposal):
    """
    Correlated polarisation/phase jump proposal. Jumps between degenerate phi/psi regions.
    """

    def __call__(self, sample, coordinates, **kwargs):
        out = copy.copy(sample)
        alpha = out['psi'] + out['phase']
        beta = out['psi'] - out['phase']

        draw = random.random()
        if draw < 0.5:
            alpha = 3.0 * np.pi * random.random()
        else:
            beta = 3.0 * np.pi * random.random() - 2 * np.pi
        out['psi'] = (alpha + beta) * 0.5
        out['phase'] = (alpha - beta) * 0.5
        return super(CorrelatedPolarisationPhaseJump, self).__call__(out)


class PolarisationPhaseJump(JumpProposal):
    """
    Correlated polarisation/phase jump proposal. Jumps between degenerate phi/psi regions.
    """

    def __call__(self, sample, coordinates, **kwargs):
        out = copy.copy(sample)
        out['phase'] += np.pi
        out['psi'] += np.pi / 2
        return super(PolarisationPhaseJump, self).__call__(out)


class DrawFlatPrior(JumpProposal):
    """
    Draws a proposal from the flattened prior distribution.
    """

    def __call__(self, sample, *args, **kwargs):
        out = copy.copy(sample)
        out = _draw_from_flat_priors(out, self.priors)
        return super(DrawFlatPrior, self).__call__(out)


class DrawApproxPrior(JumpProposal):

    def __init__(self, priors, analytic_test=True):
        """ Draws new sample from the prior distribution.

        Parameters
        ----------
        priors:
            See superclass
        analytic_test: bool, optional
            Draw from flat priors if True; Draw from defined priors if false
        """
        super(DrawApproxPrior, self).__init__(priors)
        self.analytic_test = analytic_test

    def __call__(self, sample, *args, **kwargs):
        out = copy.copy(sample)
        if self.analytic_test:
            out = _draw_from_flat_priors(out, self.priors)
        else:
            out = self.priors.sample()
            log_backward_jump = approx_log_prior(sample)
            self.log_j = log_backward_jump - approx_log_prior(out)
        return super(DrawApproxPrior, self).__call__(out)


def _draw_from_flat_priors(sample, priors):
    out = copy.copy(sample)
    for key in out.keys():
        flat_prior = Uniform(priors[key].minimum, priors[key].maximum, priors[key].name)
        out[key] = flat_prior.sample()
    return out


def approx_log_prior(sample):
    """ TODO: Make sure this was correctly translated from LALInference

    Parameters
    ----------
    sample: dict

    Returns
    -------
    An approximation for the log prior
    """
    log_p = 0
    if 'chirp_mass' in sample.keys():
        log_p += -11.0 / 6.0 * np.log(sample['chirp_mass'])

    if 'luminosity_distance' in sample.keys():
        log_p += 2 * np.log(sample['luminosity_distance'])

    if 'dec' in sample.keys():
        log_p += np.log(np.cos(sample['dec']))

    if 'tilt_1' in sample.keys():
        log_p += np.log(np.abs(np.sin(sample['tilt_1'])))

    if 'tilt_2' in sample.keys():
        log_p += np.log(np.abs(np.sin(sample['tilt_2'])))

    return log_p

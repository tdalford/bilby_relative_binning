import numpy as np

from bilby.core.utils import infer_args_from_method, infer_parameters_from_function
from bilby.core.prior.base import Prior
from bilby.core.prior.interpolated import Interped
from bilby.core.prior.analytical import DeltaFunction, PowerLaw, Uniform, LogUniform, \
    SymmetricLogUniform, Cosine, Sine, Gaussian, TruncatedGaussian, HalfGaussian, \
    LogNormal, Exponential, StudentT, Beta, Logistic, Cauchy, Gamma, ChiSquared, FermiDirac


def slab_spike_prior_factory(prior_class):

    class SlabSpikePrior(prior_class):
        def __init__(self, name=None, latex_label=None, unit=None, boundary=None, spike_loc=None,
                     spike_height=0, **additional_params):
            if 'boundary' in infer_args_from_method(super(SlabSpikePrior, self).__init__):
                super(SlabSpikePrior, self).__init__(name=name, latex_label=latex_label,
                                                     unit=unit, boundary=boundary, **additional_params)
            else:
                super(SlabSpikePrior, self).__init__(name=name, latex_label=latex_label,
                                                     unit=unit, **additional_params)

            self._required_variables = None
            self._reference_params = additional_params
            self.__class__.__name__ = 'SlabSpike{}'.format(prior_class.__name__)
            self.__class__.__qualname__ = 'SlabSpike{}'.format(prior_class.__qualname__)

            if spike_loc is None:
                self.spike_loc = self.minimum
            else:
                self.spike_loc = spike_loc
            self.spike_height = spike_height
            self.inverse_cdf_below_spike = self._find_inverse_cdf_fraction_before_spike()

        @property
        def segment_length(self):
            return self.maximum - self.minimum

        @property
        def slab_fraction(self):
            return 1 - self.spike_height

        def _find_inverse_cdf_fraction_before_spike(self):
            # binary search
            p = self.slab_fraction / 2
            for n in range(2, 100):
                print(p)
                step = self.slab_fraction * 0.5 ** n
                contracted_rescale = self._contracted_rescale(p)
                if contracted_rescale < self.spike_loc:
                    p += step
                else:
                    p -= step
            return p

        def rescale(self, val):
            val = np.atleast_1d(val)
            res = np.zeros(len(val))

            spike_start = self.inverse_cdf_below_spike

            lower_indices = np.where(val < spike_start)
            intermediate_indices = np.where(np.logical_and(val >= spike_start, val <= spike_start + self.spike_height))
            higher_indices = np.where(val > spike_start + self.spike_height)

            res[lower_indices] = self._contracted_rescale(val[lower_indices])
            res[intermediate_indices] = self.spike_loc
            res[higher_indices] = self._contracted_rescale(val[higher_indices] - self.spike_height)
            return res

        def _contracted_rescale(self, val):
            return super(SlabSpikePrior, self).rescale(val / self.slab_fraction)

        def prob(self, val):
            res = super(SlabSpikePrior, self).prob(val) * self.slab_fraction
            res[np.where(val == self.spike_loc)] = np.inf
            return res

        def ln_prob(self, val):
            res = super(SlabSpikePrior, self).ln_prob(val) + np.log(self.slab_fraction)
            res[np.where(val == self.spike_loc)] = np.inf
            return res

    return SlabSpikePrior

SlabSpikeBasePrior = slab_spike_prior_factory(Prior)  # Only for testing purposes
SlabSpikeUniform = slab_spike_prior_factory(Uniform)
SlabSpikePowerLaw = slab_spike_prior_factory(PowerLaw)
SlabSpikeGaussian = slab_spike_prior_factory(Gaussian)
SlabSpikeLogUniform = slab_spike_prior_factory(LogUniform)
SlabSpikeSymmetricLogUniform = slab_spike_prior_factory(SymmetricLogUniform)
SlabSpikeCosine = slab_spike_prior_factory(Cosine)
SlabSpikeSine = slab_spike_prior_factory(Sine)
SlabSpikeTruncatedGaussian = slab_spike_prior_factory(TruncatedGaussian)
SlabSpikeHalfGaussian = slab_spike_prior_factory(HalfGaussian)
SlabSpikeLogNormal = slab_spike_prior_factory(LogNormal)
SlabSpikeExponential = slab_spike_prior_factory(Exponential)
SlabSpikeStudentT = slab_spike_prior_factory(StudentT)
SlabSpikeBeta = slab_spike_prior_factory(Beta)
SlabSpikeLogistic = slab_spike_prior_factory(Logistic)
SlabSpikeCauchy = slab_spike_prior_factory(Cauchy)
SlabSpikeGamma = slab_spike_prior_factory(Gamma)
SlabSpikeChiSquared = slab_spike_prior_factory(ChiSquared)
SlabSpikeFermiDirac = slab_spike_prior_factory(FermiDirac)
SlabSpikeInterped = slab_spike_prior_factory(Interped)

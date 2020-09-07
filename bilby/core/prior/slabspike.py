import numpy as np

from bilby.core.prior.base import Prior
from bilby.core.utils import logger


class SlabSpikePrior(Prior):

    def __init__(self, slab, spike_loc=None, spike_height=0):
        self.slab = slab
        super().__init__(name=self.slab.name, latex_label=self.slab.latex_label, unit=self.slab.unit,
                         minimum=self.slab.minimum, maximum=self.slab.maximum,
                         check_range_nonzero=self.slab.check_range_nonzero, boundary=self.slab.boundary)
        self.spike_loc = spike_loc
        self.spike_height = spike_height
        try:
            self.inverse_cdf_below_spike = self._find_inverse_cdf_fraction_before_spike()
        except Exception as e:
            logger.warning("Disregard the following warning when running tests:\n {}".format(e))

    @property
    def spike_loc(self):
        return self._spike_loc

    @spike_loc.setter
    def spike_loc(self, spike_loc):
        if spike_loc is None:
            spike_loc = self.minimum
        if not self.minimum <= spike_loc <= self.maximum:
            raise ValueError("Spike location {} not within prior domain "
                             .format(spike_loc, self.minimum, self.maximum))
        self._spike_loc = spike_loc

    @property
    def segment_length(self):
        return self.maximum - self.minimum

    @property
    def slab_fraction(self):
        return 1 - self.spike_height

    def _find_inverse_cdf_fraction_before_spike(self):
        return float(self.slab.cdf(self.spike_loc)) * self.slab_fraction

    def rescale(self, val):
        val = np.atleast_1d(val)

        lower_indices = np.where(val < self.inverse_cdf_below_spike)[0]
        intermediate_indices = np.where(np.logical_and(
            self.inverse_cdf_below_spike <= val,
            val <= self.inverse_cdf_below_spike + self.spike_height))[0]
        higher_indices = np.where(val > self.inverse_cdf_below_spike + self.spike_height)[0]

        res = np.zeros(len(val))
        res[lower_indices] = self._contracted_rescale(val[lower_indices])
        res[intermediate_indices] = self.spike_loc
        res[higher_indices] = self._contracted_rescale(val[higher_indices] - self.spike_height)
        return res

    def _contracted_rescale(self, val):
        return self.slab.rescale(val / self.slab_fraction)

    def prob(self, val):
        res = self.slab.prob(val) * self.slab_fraction
        res = np.atleast_1d(res)
        res[np.where(val == self.spike_loc)] = np.inf
        return res

    def ln_prob(self, val):
        res = self.slab.ln_prob(val) + np.log(self.slab_fraction)
        res = np.atleast_1d(res)
        res[np.where(val == self.spike_loc)] = np.inf
        return res

    def cdf(self, val):
        res = self.slab.cdf(val) * self.slab_fraction
        res = np.atleast_1d(res)
        indices_above_spike = np.where(val > self.spike_loc)[0]
        res[indices_above_spike] += self.spike_height
        return res

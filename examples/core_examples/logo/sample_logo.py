"""
Script used to generate the samples for the bilby logo

Extra requirements
==================
- scikit-image: package for reading images into an array, available via pypi
- nestle: a pure-python nested sampling package, available via pypi

Typical run time: ~ 1 mintue
"""
import bilby
import numpy as np
import scipy.interpolate as si
from skimage import io


class Likelihood(bilby.Likelihood):
    def __init__(self, interp):
        super().__init__(parameters=dict())
        self.interp = interp

    def log_likelihood(self):
        return -1 / (self.interp(self.parameters["x"], self.parameters["y"])[0])


for letter in ["B", "I", "L", "Y"]:
    img = 1 - io.imread("{}.png".format(letter), as_gray=True)[::-1, :]
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    interp = si.interpolate.interp2d(x, y, img.T)

    likelihood = Likelihood(interp)

    priors = bilby.core.prior.PriorDict()
    priors["x"] = bilby.prior.Uniform(0, max(x), "x")
    priors["y"] = bilby.prior.Uniform(0, max(y), "y")

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="nestle",
        npoints=5000,
        label=letter,
    )
    fig = result.plot_corner(quantiles=None)

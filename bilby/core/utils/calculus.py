import numpy as np
from scipy.interpolate import interp2d
from scipy.special import logsumexp

from bilby.core.utils import logger


def derivatives(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
                epsscale=0.5, nonfixedidx=None):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.
    nonfixedidx: array_like, None
        An array of indices in `vals` that are _not_ fixed values and therefore
        can have derivatives taken. If `None` then derivatives of all values
        are calculated.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    if nonfixedidx is None:
        nonfixedidx = range(len(vals))

    if len(nonfixedidx) > len(vals):
        raise ValueError("To many non-fixed values")

    if max(nonfixedidx) >= len(vals) or min(nonfixedidx) < 0:
        raise ValueError("Non-fixed indexes contain non-existant indices")

    grads = np.zeros(len(nonfixedidx))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals) * releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps * np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps * np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in nonfixedidx:
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5 * leps  # change forwards distance to half eps
        bvals[i] -= 0.5 * leps  # change backwards distance to half eps
        cdiff = (func(fvals) - func(bvals)) / leps

        while 1:
            fvals[i] -= 0.5 * leps  # remove old step
            bvals[i] += 0.5 * leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                logger.warning("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5 * leps  # change forwards distance to half eps
            bvals[i] -= 0.5 * leps  # change backwards distance to half eps
            cdiffnew = (func(fvals) - func(bvals)) / leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff / cdiffnew)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1. - rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads


def logtrapzexp(lnf, dx):
    """
    Perform trapezium rule integration for the logarithm of a function on a regular grid.

    Parameters
    ----------
    lnf: array_like
        A :class:`numpy.ndarray` of values that are the natural logarithm of a function
    dx: Union[array_like, float]
        A :class:`numpy.ndarray` of steps sizes between values in the function, or a
        single step size value.

    Returns
    -------
    The natural logarithm of the area under the function.
    """
    return np.log(dx / 2.) + logsumexp([logsumexp(lnf[:-1]), logsumexp(lnf[1:])])


class UnsortedInterp2d(interp2d):

    def __call__(self, x, y, dx=0, dy=0, assume_sorted=False):
        """  Wrapper to scipy.interpolate.interp2d which preserves the input ordering.

        Parameters
        ----------
        x: See superclass
        y: See superclass
        dx: See superclass
        dy: See superclass
        assume_sorted: bool, optional
            This is just a place holder to prevent a warning.
            Overwriting this will not do anything

        Returns
        ----------
        array_like: See superclass

        """
        unsorted_idxs = np.argsort(np.argsort(x))
        return super(UnsortedInterp2d, self).__call__(x, y, dx=dx, dy=dy, assume_sorted=False)[unsorted_idxs]

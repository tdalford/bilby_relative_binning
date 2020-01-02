from bilby.core.prior import PriorDict
from bilby.core.utils import logger


def create_default_prior(name, default_priors_file=None):
    """Make a default prior for a parameter with a known name.

    Parameters
    ----------
    name: str
        Parameter name
    default_priors_file: str, optional
        If given, a file containing the default priors.

    Return
    ------
    prior: Prior
        Default prior distribution for that parameter, if unknown None is
        returned.
    """

    if default_priors_file is None:
        logger.debug(
            "No prior file given.")
        prior = None
    else:
        default_priors = PriorDict(filename=default_priors_file)
        if name in default_priors.keys():
            prior = default_priors[name]
        else:
            logger.debug(
                "No default prior found for variable {}.".format(name))
            prior = None
    return prior
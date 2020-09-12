import inspect
import json
import os
import shutil
from importlib import import_module

import numpy as np
import pandas as pd

from bilby.core.utils import logger, infer_args_from_method


def check_directory_exists_and_if_not_mkdir(directory):
    """ Checks if the given directory exists and creates it if it does not exist

    Parameters
    ----------
    directory: str
        Name of the directory

    """
    if directory == "":
        return
    elif not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug('Making directory {}'.format(directory))
    else:
        logger.debug('Directory {} exists'.format(directory))


class BilbyJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        from ..prior import MultivariateGaussianDist, Prior, PriorDict
        from ...gw.prior import HealPixMapPriorDist
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, PriorDict):
            return {'__prior_dict__': True, 'content': obj._get_json_dict()}
        if isinstance(obj, (MultivariateGaussianDist, HealPixMapPriorDist, Prior)):
            return {'__prior__': True, '__module__': obj.__module__,
                    '__name__': obj.__class__.__name__,
                    'kwargs': dict(obj.get_instantiation_dict())}
        try:
            from astropy import cosmology as cosmo, units
            if isinstance(obj, cosmo.FLRW):
                return encode_astropy_cosmology(obj)
            if isinstance(obj, units.Quantity):
                return encode_astropy_quantity(obj)
            if isinstance(obj, units.PrefixUnit):
                return str(obj)
        except ImportError:
            logger.debug("Cannot import astropy, cannot write cosmological priors")
        if isinstance(obj, np.ndarray):
            return {'__array__': True, 'content': obj.tolist()}
        if isinstance(obj, complex):
            return {'__complex__': True, 'real': obj.real, 'imag': obj.imag}
        if isinstance(obj, pd.DataFrame):
            return {'__dataframe__': True, 'content': obj.to_dict(orient='list')}
        if isinstance(obj, pd.Series):
            return {'__series__': True, 'content': obj.to_dict()}
        if inspect.isfunction(obj):
            return {"__function__": True, "__module__": obj.__module__, "__name__": obj.__name__}
        if inspect.isclass(obj):
            return {"__class__": True, "__module__": obj.__module__, "__name__": obj.__name__}
        return json.JSONEncoder.default(self, obj)


def encode_astropy_cosmology(obj):
    cls_name = obj.__class__.__name__
    dct = {key: getattr(obj, key) for
           key in infer_args_from_method(obj.__init__)}
    dct['__cosmology__'] = True
    dct['__name__'] = cls_name
    return dct


def encode_astropy_quantity(dct):
    dct = dict(__astropy_quantity__=True, value=dct.value, unit=str(dct.unit))
    if isinstance(dct['value'], np.ndarray):
        dct['value'] = list(dct['value'])
    return dct


def decode_astropy_cosmology(dct):
    try:
        from astropy import cosmology as cosmo
        cosmo_cls = getattr(cosmo, dct['__name__'])
        del dct['__cosmology__'], dct['__name__']
        return cosmo_cls(**dct)
    except ImportError:
        logger.debug("Cannot import astropy, cosmological priors may not be "
                     "properly loaded.")
        return dct


def decode_astropy_quantity(dct):
    try:
        from astropy import units
        if dct['value'] is None:
            return None
        else:
            del dct['__astropy_quantity__']
            return units.Quantity(**dct)
    except ImportError:
        logger.debug("Cannot import astropy, cosmological priors may not be "
                     "properly loaded.")
        return dct


def load_json(filename, gzip):
    if gzip or os.path.splitext(filename)[1].lstrip('.') == 'gz':
        import gzip
        with gzip.GzipFile(filename, 'r') as file:
            json_str = file.read().decode('utf-8')
        dictionary = json.loads(json_str, object_hook=decode_bilby_json)
    else:
        with open(filename, 'r') as file:
            dictionary = json.load(file, object_hook=decode_bilby_json)
    return dictionary


def decode_bilby_json(dct):
    if dct.get("__prior_dict__", False):
        cls = getattr(import_module(dct['__module__']), dct['__name__'])
        obj = cls._get_from_json_dict(dct)
        return obj
    if dct.get("__prior__", False):
        cls = getattr(import_module(dct['__module__']), dct['__name__'])
        obj = cls(**dct['kwargs'])
        return obj
    if dct.get("__cosmology__", False):
        return decode_astropy_cosmology(dct)
    if dct.get("__astropy_quantity__", False):
        return decode_astropy_quantity(dct)
    if dct.get("__array__", False):
        return np.asarray(dct["content"])
    if dct.get("__complex__", False):
        return complex(dct["real"], dct["imag"])
    if dct.get("__dataframe__", False):
        return pd.DataFrame(dct['content'])
    if dct.get("__series__", False):
        return pd.Series(dct['content'])
    if dct.get("__function__", False) or dct.get("__class__", False):
        default = ".".join([dct["__module__"], dct["__name__"]])
        return getattr(import_module(dct["__module__"]), dct["__name__"], default)
    return dct


def safe_file_dump(data, filename, module):
    """ Safely dump data to a .pickle file

    Parameters
    ----------
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill
        The python module to use
    """

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    shutil.move(temp_filename, filename)


def move_old_file(filename, overwrite=False):
    """ Moves or removes an old file.

    Parameters
    ----------
    filename: str
        Name of the file to be move
    overwrite: bool, optional
        Whether or not to remove the file or to change the name
        to filename + '.old'
    """
    if os.path.isfile(filename):
        if overwrite:
            logger.debug('Removing existing file {}'.format(filename))
            os.remove(filename)
        else:
            logger.debug(
                'Renaming existing file {} to {}.old'.format(filename,
                                                             filename))
            shutil.move(filename, filename + '.old')
    logger.debug("Saving result to {}".format(filename))


def safe_save_figure(fig, filename, **kwargs):
    check_directory_exists_and_if_not_mkdir(os.path.dirname(filename))
    from matplotlib import rcParams
    try:
        fig.savefig(fname=filename, **kwargs)
    except RuntimeError:
        logger.debug(
            "Failed to save plot with tex labels turning off tex."
        )
        rcParams["text.usetex"] = False
        fig.savefig(fname=filename, **kwargs)
from bilby.core.utils import infer_args_from_method

import numpy as np


def get_instantiation_dict(obj):
    subclass_args = infer_args_from_method(obj.__init__)
    dict_with_properties = get_dict_with_properties(obj)
    instantiation_dict = dict()
    for key in subclass_args:
        if isinstance(dict_with_properties[key], list):
            value = np.asarray(dict_with_properties[key]).tolist()
        else:
            value = dict_with_properties[key]
        instantiation_dict[key] = value
    return instantiation_dict


def get_dict_with_properties(obj):
    property_names = [p for p in dir(obj.__class__)
                      if isinstance(getattr(obj.__class__, p), property)]
    dict_with_properties = obj.__dict__.copy()
    for key in property_names:
        dict_with_properties[key] = getattr(obj, key)
    return dict_with_properties

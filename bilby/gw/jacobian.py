def chirp_mass_to_primary_mass(sample):
    """
    Function to map from a prior which is defined in chirp mass to primary mass

    J = dm1 / dmc = (1 + q)^(1/5) / q^(3/5)

    Parameters
    ----------
    sample: dict
        dict containing mass_ratio

    Returns
    -------
    jacobian: float
     The jacobian of the transformation
    """
    jacobian = (1 + sample['mass_ratio'])**0.2 / sample['mass_ratio']**0.6
    return jacobian


def chirp_mass_mass_ratio_to_component_masses(sample):
    """
    Function to map from a prior which is defined in chirp mass to primary mass

    J = mc / m1^2

    See (21) of https://arxiv.org/abs/1409.7215

    Parameters
    ----------
    sample: dict
        dict containing chirp_mass and mass_ratio

    Returns
    -------
    jacobian: float
     The jacobian of the transformation
    """
    mass_1 = (sample['chirp_mass'] * (1 + sample['mass_ratio'])**0.2 /
              sample['mass_ratio']**0.6)
    jacobian = sample['chirp_mass'] / mass_1**2
    return jacobian

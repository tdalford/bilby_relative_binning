=====
Bilby update to include relative binning likelihood evaluation
=====

Currently an add-on to Bilby which expands the likelihood
functionality of Bilby to include the option to use the relative
binning procedure defined in https://arxiv.org/pdf/1806.08792.pdf.

Files modified:

``/bilby/gw/likelihood.py`` (vast majority of new code, defined in RelativeBinningGravitationalWaveTransient class).

``/bilby/gw/detector/interferometer.py`` (slight function addition to get the detector response over a full frequency binning range for the relative binning technique).

``/examples/gw_examples/data_examples/relative_binning_tests.py`` (file added to test functionality of this new class and to aid in debugging.)

How to run:

A sample test is set up in the
``/examples/gw_examples/data_examples/relative_binning_tests.py`` as mentioned.


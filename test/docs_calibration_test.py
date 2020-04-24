import doctest
import unittest

from bilby.gw import calibration

# Doctests
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(calibration))
    return tests



if __name__ == '__main__':
    unittest.main()

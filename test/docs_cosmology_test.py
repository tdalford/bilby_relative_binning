import doctest
import unittest

from bilby.gw import cosmology

# Doctests
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(cosmology))
    return tests



if __name__ == '__main__':
    unittest.main()

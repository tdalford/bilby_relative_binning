import doctest
import unittest

from bilby.core import utils

# Doctests
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(utils))
    return tests



if __name__ == '__main__':
    unittest.main()

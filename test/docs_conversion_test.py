import doctest
import unittest

from bilby.gw import conversion

# Doctests
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(conversion))
    return tests



if __name__ == '__main__':
    unittest.main()

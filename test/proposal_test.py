import unittest
import mock
import random
import numpy as np
from bilby.core import prior
from bilby.core.sampler import proposal


class TestJumpProposal(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, boundary='reflecting'),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary='periodic'),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.sample_above = dict(reflecting=1.1, periodic=1.1, default=1.1)
        self.sample_below = dict(reflecting=-0.6, periodic=-0.6, default=-0.6)
        self.jump_proposal = proposal.JumpProposal(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.sample_above
        del self.jump_proposal

    def set_get_log_j(self):
        self.jump_proposal.log_j = 2.3
        self.assertEqual(2.3, self.jump_proposal.log_j)

    def test_boundary_above_reflecting(self):
        new_sample = self.jump_proposal(self.sample_above)
        self.assertAlmostEqual(0.9, new_sample['reflecting'])

    def test_boundary_above_periodic(self):
        new_sample = self.jump_proposal(self.sample_above)
        self.assertAlmostEqual(-0.4, new_sample['periodic'])

    def test_boundary_above_default(self):
        new_sample = self.jump_proposal(self.sample_above)
        self.assertAlmostEqual(1.1, new_sample['default'])

    def test_boundary_below_reflecting(self):
        new_sample = self.jump_proposal(self.sample_below)
        self.assertAlmostEqual(-0.4, new_sample['reflecting'])

    def test_boundary_below_periodic(self):
        new_sample = self.jump_proposal(self.sample_below)
        self.assertAlmostEqual(0.9, new_sample['periodic'])

    def test_boundary_below_default(self):
        new_sample = self.jump_proposal(self.sample_below)
        self.assertAlmostEqual(-0.6, new_sample['default'])

    def test_priors(self):
        self.assertEqual(self.priors, self.jump_proposal.priors)


class TestNormJump(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, boundary='reflecting'),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary='periodic'),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.jump_proposal = proposal.NormJump(step_size=3.0, priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_step_size_init(self):
        self.assertEqual(3.0, self.jump_proposal.step_size)

    def test_set_step_size(self):
        self.jump_proposal.step_size = 1.0
        self.assertEqual(1.0, self.jump_proposal.step_size)

    def test_jump_proposal_call(self):
        with mock.patch("numpy.random.normal") as m:
            m.return_value = 0.5
            sample = dict(reflecting=0.0, periodic=0.0, default=0.0)
            new_sample = self.jump_proposal(sample)
            expected = dict(reflecting=0.5, periodic=0.5, default=0.5)
            self.assertDictEqual(expected, new_sample)


class TestEnsembleWalk(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, boundary='reflecting'),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary='periodic'),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.jump_proposal = proposal.EnsembleWalk(random_number_generator=random.random,
                                                   n_points=4, priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_n_points_init(self):
        self.assertEqual(4, self.jump_proposal.n_points)

    def test_set_n_points(self):
        self.jump_proposal.n_points = 3
        self.assertEqual(3, self.jump_proposal.n_points)

    def test_random_number_generator_init(self):
        self.assertEqual(random.random, self.jump_proposal.random_number_generator)

    def test_get_center_of_mass(self):
        samples = [dict(reflecting=0.1*i, periodic=0.1*i, default=0.1*i)
                   for i in range(3)]
        expected = dict(reflecting=0.1, periodic=0.1, default=0.1)
        actual = self.jump_proposal.get_center_of_mass(samples)
        for key in samples[0].keys():
            self.assertAlmostEqual(expected[key], actual[key])

    # def test_jump_proposal_call(self):
    #     with mock.patch('random.random') as m1:
    #         with mock.patch('functools.reduce') as m2:
    #             with mock.patch('random.sample') as m3:
    #                 m1.return_value = 2.0
    #                 m2.return_value = 0.0
    #                 m3.return_value = np.array([0.1, 0.1, 0.1])
    #                 sample = np.array([0.1, 0.1, 0.1])
    #                 new_sample = self.jump_proposal(sample, coordinates=None)
    #                 expected = np.array([0.2, 0.2, 0.2])
    #                 self.assertTrue(np.allclose(expected, new_sample))

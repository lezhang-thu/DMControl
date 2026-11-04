"""Ensure ensemble dynamics participate in SPR's module lifecycle."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spr_sac import SPRSacAgent
from transition_model import EnsembleOfProbabilisticTransitionModels


class EnsembleTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_ensemble_members_follow_dtype_and_state_dict(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = EnsembleOfProbabilisticTransitionModels(4, (2,), 8, 3)
        model = model.to(dtype=torch.float64)
        self.assertEqual(len(model.state_dict()), 3 * len(model.models[0].state_dict()))
        self.assertTrue(all(parameter.dtype == torch.float64
                            for parameter in model.parameters()))
        mus, sigmas = model(torch.zeros(2, 6, dtype=torch.float64))
        self.assertEqual(mus.shape, (3, 2, 4))
        self.assertEqual(sigmas.shape, mus.shape)
        model.eval()
        self.assertTrue(all(not member.training for member in model.models))

    def test_spr_optimizer_updates_registered_ensemble_dynamics(self):
        torch.manual_seed(31)
        with contextlib.redirect_stdout(io.StringIO()):
            agent = SPRSacAgent(
                obs_shape=(3, 84, 84), action_shape=(2,), device='cpu',
                augmentation=('none',), transition_model_type='ensemble',
                transition_model_layer_width=8, encoder_feature_dim=4,
                latent_dim=8, hidden_dim=8, num_layers=2, num_filters=2,
                jumps=1, imag_horizon=1,
            )
        ensemble = agent.SPR.transition_model
        optimized = {id(parameter) for group in agent.spr_optimizer.param_groups
                     for parameter in group['params']}
        self.assertTrue(all(id(parameter) in optimized for parameter in ensemble.parameters()))
        self.assertTrue(any(name.startswith('transition_model.models.')
                            for name in agent.SPR.state_dict()))
        before = ensemble.models[0].fc.weight.detach().clone()
        sample = {
            'observation': torch.randint(0, 256, (2, 2, 3, 1, 84, 84)).float(),
            'action': torch.rand(2, 2, 2) * 2 - 1,
        }
        with mock.patch('transition_model.random.choice', return_value=ensemble.models[0]):
            agent.update_spr(sample, mock.Mock(), step=1)
        self.assertFalse(torch.equal(before, ensemble.models[0].fc.weight))


if __name__ == '__main__':
    unittest.main()

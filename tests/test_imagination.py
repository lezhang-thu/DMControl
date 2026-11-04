"""Checks for detached world-model readouts and continuous on-policy updates."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spr_sac import SPRSacAgent, imagined_lambda_return, imagined_reach_weights


class RecordingLogger:
    def __init__(self):
        self.metrics = {}

    def log(self, key, value, step):
        self.metrics[key] = float(value.detach() if isinstance(value, torch.Tensor) else value)


def make_agent(**kwargs):
    settings = dict(
        obs_shape=(3, 84, 84), action_shape=(2,), device='cpu',
        augmentation=('none',), transition_model_type='deterministic',
        transition_model_layer_width=16, encoder_feature_dim=8,
        latent_dim=16, hidden_dim=16, num_layers=2, num_filters=4,
        jumps=2, log_interval=1, actor_update_freq=1,
        critic_target_update_freq=1, imag_horizon=3,
        reward_loss_weight=1.0, continue_loss_weight=1.0,
    )
    settings.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return SPRSacAgent(**settings)


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


class ImaginationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(19)

    def assert_unchanged(self, module, before):
        for name, parameter in module.named_parameters():
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def assert_changed(self, module, before):
        self.assertTrue(any(not torch.equal(parameter, before[name])
                            for name, parameter in module.named_parameters()))

    def test_soft_lambda_returns_match_hand_computed_targets(self):
        rewards = torch.tensor([1., 2.]).view(2, 1, 1)
        continues = torch.ones_like(rewards)
        values = torch.tensor([10., 20., 30.]).view(3, 1, 1)
        log_probs = torch.tensor([-.5, -1., -2.]).view(3, 1, 1)
        cases = [(0., [19.18, 29.36]), (.5, [23.392, 29.36]),
                 (1., [27.604, 29.36])]
        for lamb, expected in cases:
            with self.subTest(lamb=lamb):
                result = imagined_lambda_return(
                    rewards, continues, values, log_probs, .9, lamb, .2)
                torch.testing.assert_close(result.flatten(), torch.tensor(expected))

    def test_terminal_transition_stops_return_and_reach_weights(self):
        rewards = torch.tensor([1., 2., 100.]).view(3, 1, 1)
        continues = torch.tensor([1., 0., .5]).view(3, 1, 1)
        values = torch.full((4, 1, 1), 500.)
        result = imagined_lambda_return(
            rewards, continues, values, torch.zeros_like(values), .9, 1., 0.)
        torch.testing.assert_close(result[:2].flatten(), torch.tensor([2.8, 2.]))
        weights = imagined_reach_weights(continues, .9)
        torch.testing.assert_close(weights.flatten(), torch.tensor([1., .9, 0.]))

    def test_outcome_heads_cut_the_gradient_to_their_inputs(self):
        agent = make_agent()
        latent = torch.randn(4, 8, requires_grad=True)
        reward, logits = agent.SPR.predict_outcomes(latent)
        (reward.square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits))).backward()
        self.assertIsNone(latent.grad)
        self.assertTrue(any(p.grad is not None for p in agent.SPR.parameters()))
        for module in (agent.SPR.encoder, agent.SPR.transition_model):
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_head_supervision_changes_readouts_without_changing_world_model(self):
        agent = make_agent()
        observations = torch.randint(0, 256, (4, 3, 84, 84)).float()
        next_observations = torch.randint(0, 256, observations.shape).float()
        actions = torch.rand(4, 2) * 2 - 1
        rewards = torch.tensor([[1.], [3.], [2.], [4.]])
        not_done = torch.tensor([[1.], [0.], [1.], [0.]])
        frozen = [agent.SPR.encoder, agent.SPR.transition_model,
                  agent.SPR.target_encoder, agent.critic.Q1, agent.actor.trunk]
        before = [snapshot(module) for module in frozen]
        probe = torch.randn(4, 8)
        outcomes_before = [value.detach().clone()
                           for value in agent.SPR.predict_outcomes(probe)]
        logger = RecordingLogger()
        agent.update_world_heads(observations, actions, rewards,
                                 next_observations, not_done, logger, 1)
        for module, parameters in zip(frozen, before):
            self.assert_unchanged(module, parameters)
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        for previous, current in zip(outcomes_before, agent.SPR.predict_outcomes(probe)):
            self.assertFalse(torch.equal(previous, current.detach()))
        self.assertTrue(logger.metrics)
        self.assertTrue(all(torch.isfinite(torch.tensor(value))
                            for value in logger.metrics.values()))

    def test_imagination_is_detached_and_actions_are_continuous_and_bounded(self):
        agent = make_agent()
        observations = torch.randint(0, 256, (4, 3, 84, 84)).float()
        rollout = agent.imagine(observations)
        expected_shapes = {
            'latents': (4, 4, 8), 'actions': (3, 4, 2), 'pre_tanh': (3, 4, 2),
            'rewards': (3, 4, 1), 'continues': (3, 4, 1),
            'values': (4, 4, 1), 'log_probs': (4, 4, 1),
        }
        for key, shape in expected_shapes.items():
            with self.subTest(key=key):
                self.assertEqual(tuple(rollout[key].shape), shape)
                self.assertFalse(rollout[key].requires_grad)
                self.assertTrue(torch.isfinite(rollout[key]).all())
        self.assertTrue((rollout['actions'].abs() <= 1).all())
        torch.testing.assert_close(rollout['actions'], rollout['pre_tanh'].tanh())
        self.assertTrue(((rollout['continues'] >= 0) &
                         (rollout['continues'] <= 1)).all())

    def test_imagination_updates_actor_and_q_but_not_spr_or_temperature(self):
        agent = make_agent()
        observations = torch.randint(0, 256, (4, 3, 84, 84)).float()
        spr_before = snapshot(agent.SPR)
        actor_before = snapshot(agent.actor.trunk)
        q1_before, q2_before = snapshot(agent.critic.Q1), snapshot(agent.critic.Q2)
        alpha_before = agent.log_alpha.detach().clone()
        target_before = snapshot(agent.critic_target)
        agent.update_imagination(observations, RecordingLogger(), 1, scale=1.)
        self.assert_unchanged(agent.SPR, spr_before)
        self.assert_unchanged(agent.critic_target, target_before)
        torch.testing.assert_close(agent.log_alpha, alpha_before, rtol=0, atol=0)
        self.assert_changed(agent.actor.trunk, actor_before)
        self.assert_changed(agent.critic.Q1, q1_before)
        self.assert_changed(agent.critic.Q2, q2_before)
        self.assertTrue(all(p.grad is None for p in agent.SPR.parameters()))

    def test_policy_and_world_model_share_the_same_latent_coordinates(self):
        agent = make_agent()
        observations = torch.randint(0, 256, (3, 3, 84, 84)).float()
        with torch.no_grad():
            torch.testing.assert_close(agent.actor.encoder(observations),
                                       agent.SPR.encoder(observations), rtol=0, atol=0)
        self.assertIs(agent.actor.encoder.fc.weight, agent.critic.encoder.fc.weight)
        self.assertIs(agent.actor.encoder.ln.weight, agent.critic.encoder.ln.weight)

    def test_score_function_treats_a_reparameterized_sample_as_constant(self):
        agent = make_agent()
        latent = torch.zeros(1, 8)
        mu, log_std = agent.actor.distribution_parameters(latent)
        noise = torch.tensor([[1., -.5]])
        sample = mu + log_std.exp() * noise
        sample.retain_grad()
        expected_mu_gradient = (noise / log_std.exp()).detach().squeeze(0)
        agent.actor.log_prob_from_latent(latent, sample).sum().backward()
        self.assertIsNone(sample.grad)
        torch.testing.assert_close(agent.actor.trunk[-1].bias.grad[:2],
                                   expected_mu_gradient)

    def test_action_dependent_q_is_not_used_as_the_policy_baseline(self):
        # Exact Q(s,a)=r(a)=a[0]: using the sampled action's Q as a baseline
        # would cancel every advantage and lose the reward-increasing gradient.
        agent = make_agent(imag_horizon=1, imag_actor_weight=1., imag_critic_weight=0.)
        latent = torch.zeros(2, 8)
        with torch.no_grad():
            _, log_std = agent.actor.distribution_parameters(latent)
            samples = torch.tensor([[1., 0.], [-1., 0.]]) * log_std.exp()
            actions = samples.tanh()
        rewards = actions[:, :1].unsqueeze(0)
        rollout = dict(
            latents=torch.stack([latent, latent]), actions=actions.unsqueeze(0),
            pre_tanh=samples.unsqueeze(0), rewards=rewards,
            continues=torch.zeros_like(rewards),
            values=torch.cat([rewards, torch.zeros_like(rewards)]),
            log_probs=torch.zeros(2, 2, 1))

        def independent_mean_sample(features):
            # Fix the independent baseline at the mean action and suppress the
            # entropy gradient, leaving only the reward's score-function signal.
            action = features.new_zeros((features.shape[0], 2))
            return action, action, action[:, :1], action

        with mock.patch.object(agent, 'imagine', return_value=rollout), \
                mock.patch.object(agent.actor, 'forward_latent',
                                  side_effect=independent_mean_sample), \
                mock.patch.object(agent.critic_target.Q1, 'forward',
                                  side_effect=lambda features, action: action[:, :1]), \
                mock.patch.object(agent.critic_target.Q2, 'forward',
                                  side_effect=lambda features, action: action[:, :1]):
            agent.update_imagination(torch.empty(0), RecordingLogger(), 1)
        self.assertLess(float(agent.actor.trunk[-1].bias.grad[0]), -.5)
        self.assertGreater(float(agent.actor.trunk[-1].bias.detach()[0]), 0.)

    def test_one_time_warmup_has_hold_then_ramp_and_never_restarts(self):
        agent = make_agent(imag_warmup=4)
        self.assertEqual(agent.model_updates, 0)
        for count, expected in ((0, 0.), (3, 0.), (4, 0.), (5, .25),
                                (6, .5), (8, 1.), (100000, 1.)):
            with self.subTest(model_updates=count):
                agent.model_updates = count
                self.assertEqual(agent.imagination_scale, expected)

    def test_live_updates_wait_for_successful_spr_training_before_imagination(self):
        agent = make_agent(imag_warmup=2)
        observations = torch.randint(0, 256, (3, 2, 3, 1, 84, 84)).float()
        actions = torch.rand(3, 2, 2) * 2 - 1
        rewards = torch.ones(3, 2, 1)
        spr_samples = dict(observation=observations, action=actions, reward=rewards)
        real_batch = (observations[0].flatten(1, 2), actions[0], rewards[0],
                      observations[1].flatten(1, 2), torch.ones(2, 1))
        replay = mock.Mock()
        replay.sample_spr.return_value = (*real_batch, None)
        logger = RecordingLogger()
        with mock.patch.object(agent, 'imagine', wraps=agent.imagine) as imagine:
            agent.update(replay, logger, step=1)
            self.assertEqual(agent.model_updates, 0)
            imagine.assert_not_called()
            replay.sample_spr.return_value = (*real_batch, spr_samples)
            for count, scale in ((1, 0.), (2, 0.), (3, .5), (4, 1.)):
                agent.update(replay, logger, step=count + 1)
                self.assertEqual(agent.model_updates, count)
                self.assertEqual(agent.imagination_scale, scale)
                self.assertEqual(imagine.call_count, max(0, count - 2))


if __name__ == '__main__':
    unittest.main()

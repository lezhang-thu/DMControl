# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomCrop

import utils
from encoder import PixelEncoder
from transition_model import DeterministicTransitionModel


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers,
                 num_filters):
        super().__init__()

        self.encoder = PixelEncoder(obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]))

        self.apply(weight_init)

    def forward(self,
                obs,
                compute_pi=True,
                compute_log_pi=True,
                detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (log_std + 1)


        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std



class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(obs_dim + action_dim,
                                             hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = PixelEncoder(obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0],
                            hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0],
                            hidden_dim)

        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)


        return q1, q2



class SPR(nn.Module):
    """Predict future image representations with a latent transition model."""
    def __init__(self,
                 action_shape,
                 critic,
                 transition_model_layer_width=512,
                 encoder_feature_dim=50,
                 jumps=5,
                 latent_dim=512,
                 momentum_tau=1.0):
        super(SPR, self).__init__()
        self.jumps = jumps
        self.momentum_tau = momentum_tau
        self.transition_model = DeterministicTransitionModel(
            encoder_feature_dim, action_shape, transition_model_layer_width)
        self.encoder = critic.encoder
        self.target_encoder = copy.deepcopy(critic.encoder)
        self.global_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_target_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_final_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.crop = RandomCrop(
            size=self.encoder.obs_shape[-2:], p=1.0, cropping_mode="slice")

    def global_spr_loss(self, latents, target_latents):
        global_latents = self.global_classifier(latents)  # proj
        global_latents = self.global_final_classifier(global_latents)  # pred
        with torch.no_grad():
            global_targets = self.global_target_classifier(target_latents)
        return self.spr_loss(global_latents, global_targets)

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        return F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean()

    @torch.no_grad()
    def augment(self, images):
        # PixelEncoder normalizes observations; keep their 0..255 scale here.
        flat_images = images.float().reshape(-1, *images.shape[-3:])
        cropped = self.crop(flat_images)
        noise = torch.randn((cropped.size(0), 1, 1, 1), device=cropped.device)
        augmented = cropped * (1.0 + 0.05 * noise.clamp(-2.0, 2.0))
        return augmented.reshape(*images.shape[:-3], *augmented.shape[1:])

    def do_spr_loss(self, pred_latents, observation):
        latents = torch.stack(pred_latents, dim=1).flatten(0, 1)
        target_images = observation[1:self.jumps + 1]
        target_images = target_images.transpose(0, 1).flatten(2, 3)
        target_images = self.augment(target_images)

        with torch.no_grad():
            target_latents = self.target_encoder(target_images.flatten(0, 1))

        return self.global_spr_loss(latents, target_latents)


class SPRSacAgent(object):
    """SPR representation learning with SAC."""
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            # from spr
            transition_model_layer_width=512,
            jumps=5,
            latent_dim=512,
            momentum_tau=1.0,
            model_spr_weight=5.0,
            auxiliary_task_lr=1e-3,
            # sac
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            alpha_beta=0.9,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            log_interval=100):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]

        self.jumps = jumps
        self.model_spr_weight = model_spr_weight

        self.actor = Actor(obs_shape, action_shape, hidden_dim,
                           encoder_feature_dim, actor_log_std_min,
                           actor_log_std_max, num_layers,
                           num_filters).to(device)

        self.critic = Critic(obs_shape, action_shape, hidden_dim,
                             encoder_feature_dim, num_layers,
                             num_filters).to(device)

        self.critic_target = Critic(obs_shape, action_shape, hidden_dim,
                                    encoder_feature_dim,
                                    num_layers, num_filters).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=(actor_beta, 0.999))

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=(critic_beta, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=(alpha_beta, 0.999))

        self.SPR = SPR(
            action_shape, self.critic,
            transition_model_layer_width=transition_model_layer_width,
            encoder_feature_dim=encoder_feature_dim, jumps=jumps,
            latent_dim=latent_dim, momentum_tau=momentum_tau,
        ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=2 * encoder_lr)
        self.spr_optimizer = torch.optim.Adam(self.SPR.parameters(), lr=auxiliary_task_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.SPR.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs,
                                     compute_pi=False,
                                     compute_log_pi=False)
            return mu.detach().cpu().numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.detach().cpu().numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_spr(self, spr_kwargs, L, step):
        if self.jumps == 0:
            if step % self.log_interval == 0:
                L.log('train/spr_loss', 0.0, step)
            return

        # [1+self.jumps, B, 9, 1, 84, 84]
        observation = spr_kwargs["observation"]
        # [1+self.jumps, B, dim_A]
        action = spr_kwargs["action"]

        input_obs = observation[0].flatten(1, 2)
        input_obs = self.SPR.augment(input_obs)
        latent = self.SPR.encoder(input_obs)

        pred_latents = []
        for step_action in action[:self.jumps]:
            latent = self.SPR.transition_model(
                torch.cat([latent, step_action], dim=1))
            pred_latents.append(latent)

        spr_loss = self.model_spr_weight * self.SPR.do_spr_loss(
            pred_latents, observation)

        self.encoder_optimizer.zero_grad()
        self.spr_optimizer.zero_grad()
        spr_loss.backward()
        self.encoder_optimizer.step()
        self.spr_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/spr_loss', spr_loss, step)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, spr_kwargs = replay_buffer.sample_spr()

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_spr(spr_kwargs, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.encoder,
                                     self.critic_target.encoder,
                                     self.encoder_tau)
            utils.soft_update_params(self.SPR.encoder, self.SPR.target_encoder,
                                     self.SPR.momentum_tau)
            utils.soft_update_params(self.SPR.global_classifier,
                                     self.SPR.global_target_classifier,
                                     self.SPR.momentum_tau)

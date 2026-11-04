# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import (CenterCrop, RandomAffine, RandomCrop,
                                 RandomResizedCrop)
from kornia.filters import GaussianBlur2d

import utils
from encoder import make_encoder
from transition_model import make_transition_model


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


def imagined_lambda_return(rewards, continues, values, log_probs, discount,
                           lamb, alpha):
    """Soft lambda targets, time-major, with arrival rewards/continuations.

    ``values`` are sampled target Q(s_t, a_t), including the final bootstrap.
    Future policy entropy is included exactly once at every transition, so
    horizon=1 agrees with the real SAC critic target.
    """
    returns = []
    value = values[-1]
    for t in reversed(range(rewards.shape[0])):
        value = rewards[t] + discount * continues[t] * (
            (1 - lamb) * values[t + 1] + lamb * value
            - alpha * log_probs[t + 1])
        returns.append(value)
    return torch.stack(returns[::-1])


def imagined_reach_weights(continues, discount):
    """Probability/discount of reaching each action state; w(s_0) = 1."""
    return torch.cat([torch.ones_like(continues[:1]),
                      (discount * continues[:-1]).cumprod(dim=0)], dim=0)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers,
                 num_filters):
        super().__init__()

        self.encoder = make_encoder(encoder_type,
                                    obs_shape,
                                    encoder_feature_dim,
                                    num_layers,
                                    num_filters,
                                    output_logits=True)

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

        return self.forward_latent(obs, compute_pi, compute_log_pi)

    def distribution_parameters(self, latent):
        mu, log_std = self.trunk(latent).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (log_std + 1)

        return mu, log_std

    def forward_latent(self, latent, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.distribution_parameters(latent)
        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log_prob_from_latent(self, latent, pre_tanh):
        mu, log_std = self.distribution_parameters(latent)
        sample = pre_tanh.detach()
        log_prob = gaussian_logprob((sample - mu) / log_std.exp(), log_std)
        correction = 2 * (np.log(2) - sample - F.softplus(-2 * sample))
        return log_prob - correction.sum(-1, keepdim=True)



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
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = make_encoder(encoder_type,
                                    obs_shape,
                                    encoder_feature_dim,
                                    num_layers,
                                    num_filters,
                                    output_logits=True)

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



class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class SPR(nn.Module):
    """Some Information about SPR"""
    def __init__(self,
                 action_shape,
                 critic,
                 critic_target,
                 augmentation,
                 transition_model_type='probabilistic',
                 transition_model_layer_width=512,
                 encoder_feature_dim=50,
                 jumps=5,
                 latent_dim=512,
                 time_offset=0,
                 momentum_tau=1.0,
                 aug_prob=1.0,
                 output_type="continuous"):
        super(SPR, self).__init__()
        self.jumps = jumps
        self.time_offset = time_offset
        self.momentum_tau = momentum_tau
        self.aug_prob = aug_prob
        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape,
            transition_model_layer_width)
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
        self.transforms = nn.ModuleList()
        self.eval_transforms = nn.ModuleList()
        self.uses_augmentation = True
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(degrees=5, translate=(.14, .14),
                                              scale=(.9, 1.1), shear=(-5, 5), p=0.5)
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop(size=(84, 84), p=1.0, cropping_mode="slice")
                eval_transformation = CenterCrop(size=(84, 84), p=1.0, cropping_mode="slice")
                self.uses_augmentation = True
            elif aug == "rrc":
                transformation = RandomResizedCrop(size=(84, 84), scale=(0.8, 1.0),
                                                   p=1.0, cropping_mode="resample")
                eval_transformation = CenterCrop(size=(84, 84), p=1.0)
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4),
                                               RandomCrop(size=(84, 84), p=1.0, cropping_mode="slice"))
                eval_transformation = CenterCrop(size=(84, 84), p=1.0)
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        self.reward_predictor = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 1))
        self.continue_predictor = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 1))
        nn.init.zeros_(self.reward_predictor[-1].weight)
        nn.init.zeros_(self.reward_predictor[-1].bias)

    def predict_outcomes(self, latent):
        # Heads are readouts: neither supervised loss may shape the world model.
        latent = latent.detach()
        return self.reward_predictor(latent), self.continue_predictor(latent)

    def global_spr_loss(self, latents, target_latents, observation):
        global_latents = self.global_classifier(latents)  # proj
        global_latents = self.global_final_classifier(global_latents)  # pred
        with torch.no_grad():
            global_targets = self.global_target_classifier(target_latents)
        targets = global_targets.reshape(-1, observation.shape[1], self.jumps + 1,
                                      global_targets.shape[-1]).transpose(
                                          1, 2)
        latents = global_latents.reshape(-1, observation.shape[1], self.jumps + 1,
                                      global_latents.shape[-1]).transpose(
                                          1, 2)
        loss = self.spr_loss(latents, targets)
        return loss

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = utils.maybe_transform(image,
                                              transform,
                                              eval_transform,
                                              p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        # PixelEncoder normalizes 0..255 observations; preserve that scale.
        images = images.float()
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None, flat_images)
        processed_images = processed_images.reshape(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    def do_spr_loss(self, pred_latents, observation):
        pred_latents = torch.stack(pred_latents, 1)
        latents = pred_latents[:observation.shape[1]].flatten(
            0, 1)  # batch*jumps, *
        neg_latents = pred_latents[observation.shape[1]:].flatten(0, 1)
        latents = torch.cat([latents, neg_latents], 0)
        target_images = observation[self.time_offset:self.jumps +
                                    self.time_offset + 1]
        target_images = target_images.transpose(0, 1).flatten(2, 3)
        target_images = self.transform(target_images, True)  # Augment

        with torch.no_grad():
            target_latents = self.target_encoder(target_images.flatten(0, 1))

        global_loss = self.global_spr_loss(latents, target_latents,
                                           observation)
        # split to batch, jumps
        spr_loss = global_loss.reshape(-1, observation.shape[1])
        return spr_loss


class SPRSacAgent(object):
    """SPR representation learning with SAC."""
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            # from spr
            augmentation=(),
            transition_model_type='probabilistic',
            transition_model_layer_width=512,
            jumps=5,
            latent_dim=512,
            time_offset=0,
            momentum_tau=1.0,
            aug_prob=1.0,
            t0_spr_loss_weight=0.0,
            model_spr_weight=5.0,
            reward_loss_weight=1.0,
            continue_loss_weight=1.0,
            imag_horizon=5,
            imag_warmup=2000,
            imag_lambda=0.95,
            imag_actor_weight=0.1,
            imag_critic_weight=0.05,
            imag_batch_size=128,
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
            encoder_type='pixel',
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            log_interval=100,
            detach_encoder=False):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.jumps = jumps
        self.model_spr_weight = model_spr_weight
        self.t0_spr_loss_weight = t0_spr_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.continue_loss_weight = continue_loss_weight
        if imag_horizon < 0 or imag_warmup < 0 or imag_batch_size < 1:
            raise ValueError("Imagination horizon/warmup must be nonnegative and batch size positive")
        if not 0 <= imag_lambda <= 1:
            raise ValueError("imag_lambda must be in [0, 1]")
        if min(reward_loss_weight, continue_loss_weight,
               imag_actor_weight, imag_critic_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if imag_horizon and (encoder_type != 'pixel' or jumps < 1):
            raise ValueError("Imagination requires the pixel SPR model with jumps >= 1")
        self.imag_horizon = imag_horizon
        self.imag_warmup = imag_warmup
        self.imag_lambda = imag_lambda
        self.imag_actor_weight = imag_actor_weight
        self.imag_critic_weight = imag_critic_weight
        self.imag_batch_size = imag_batch_size
        self.model_updates = 0
        self.return_quantiles = None

        self.actor = Actor(obs_shape, action_shape, hidden_dim, encoder_type,
                           encoder_feature_dim, actor_log_std_min,
                           actor_log_std_max, num_layers,
                           num_filters).to(device)

        self.critic = Critic(obs_shape, action_shape, hidden_dim, encoder_type,
                             encoder_feature_dim, num_layers,
                             num_filters).to(device)

        self.critic_target = Critic(obs_shape, action_shape, hidden_dim,
                                    encoder_type, encoder_feature_dim,
                                    num_layers, num_filters).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        if self.imag_horizon:
            # Real and imagined policies must consume the same SPR latent space.
            self.actor.encoder = self.critic.encoder
        else:
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        actor_parameters = (self.actor.trunk.parameters() if self.imag_horizon
                            else self.actor.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_parameters,
                                                lr=actor_lr,
                                                betas=(actor_beta, 0.999))

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=(critic_beta, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=(alpha_beta, 0.999))

        if self.encoder_type == 'pixel':
            # create SPR representation model
            self.SPR = SPR(action_shape,
                           self.critic,
                           self.critic_target,
                           augmentation,
                           transition_model_type,
                           transition_model_layer_width,
                           encoder_feature_dim,
                           jumps,
                           latent_dim,
                           time_offset,
                           momentum_tau,
                           aug_prob,
                           output_type='continuous').to(self.device)
            # optimizer for critic encoder and spr loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=2 * encoder_lr)
            # Dedicated head optimizer, disjoint from the original SPR optimizer.
            head_parameters = list(self.SPR.reward_predictor.parameters()) + list(
                self.SPR.continue_predictor.parameters())
            head_ids = {id(parameter) for parameter in head_parameters}
            self.spr_optimizer = torch.optim.Adam(
                [p for p in self.SPR.parameters() if id(p) not in head_ids],
                lr=auxiliary_task_lr)
            self.world_head_optimizer = torch.optim.Adam(
                head_parameters, lr=auxiliary_task_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
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
        if self.encoder_type == 'pixel' and obs.shape[-1] != self.image_size:
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
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
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
        if self.imag_horizon:
            with torch.no_grad():
                latent = self.critic.encoder(obs)
            _, pi, log_pi, log_std = self.actor.forward_latent(latent)
            actor_Q1 = self.critic.Q1(latent, pi)
            actor_Q2 = self.critic.Q2(latent, pi)
        else:
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
        # [1+self.jumps, B, 9, 1, 84, 84]
        observation = spr_kwargs["observation"]
        # [1+self.jumps, B, dim_A]
        action = spr_kwargs["action"]
        pred_latents = []
        input_obs = self.SPR.transform(observation[0].flatten(1, 2), augment=True)
        latent = self.SPR.encoder(input_obs)
        pred_latents.append(latent)
        for j in range(self.jumps):
            latent = self.SPR.transition_model.sample_prediction(
                torch.cat([latent, action[j]], dim=1))
            pred_latents.append(latent)

        spr_loss = self.SPR.do_spr_loss(pred_latents, observation)
        if self.jumps > 0:
            model_spr_loss = spr_loss[1:].mean(0)  # (jumps, bs) => (bs, )
            t0_spr_loss = spr_loss[0]
        else:
            t0_spr_loss = spr_loss[0]
            model_spr_loss = torch.zeros_like(spr_loss)

        final_spr_loss = self.t0_spr_loss_weight * t0_spr_loss.mean() + \
            self.model_spr_weight * model_spr_loss.mean()

        self.encoder_optimizer.zero_grad()
        self.spr_optimizer.zero_grad()
        final_spr_loss.backward()
        self.encoder_optimizer.step()
        self.spr_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/spr_loss', final_spr_loss, step)

    @property
    def imagination_scale(self):
        # A single initial warmup and ramp, counted in successful SPR updates.
        if not self.imag_horizon:
            return 0.0
        if self.imag_warmup == 0:
            return 1.0
        return float(np.clip((self.model_updates - self.imag_warmup) /
                             self.imag_warmup, 0.0, 1.0))

    def update_world_heads(self, obs, action, reward, next_obs, not_done, L, step):
        # Use unrestricted real transitions, including true terminations. The
        # SPR sequence sampler excludes boundaries and cannot teach c=0 alone.
        with torch.no_grad():
            latent = self.SPR.encoder(obs)
            predicted_next = self.SPR.transition_model.sample_prediction(
                torch.cat([latent, action], dim=-1))
            real_next = self.SPR.target_encoder(next_obs)
        pred_reward, pred_continue = self.SPR.predict_outcomes(predicted_next)
        real_reward, real_continue = self.SPR.predict_outcomes(real_next)
        reward_loss = (F.mse_loss(pred_reward, reward) +
                       F.mse_loss(real_reward, reward))
        continue_loss = (F.binary_cross_entropy_with_logits(pred_continue, not_done) +
                         F.binary_cross_entropy_with_logits(real_continue, not_done))
        loss = (self.reward_loss_weight * reward_loss +
                self.continue_loss_weight * continue_loss)
        self.world_head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_head_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/reward_loss', reward_loss, step)
            L.log('train/continue_loss', continue_loss, step)

    @torch.no_grad()
    def imagine(self, obs):
        """Fresh on-policy trajectories; no graph through the world model."""
        latent = self.SPR.encoder(obs[:self.imag_batch_size])
        latents, actions, pre_tanh, rewards, continues, values, log_probs = (
            [] for _ in range(7))
        for t in range(self.imag_horizon + 1):
            latents.append(latent)
            mu, log_std = self.actor.distribution_parameters(latent)
            sample = mu + log_std.exp() * torch.randn_like(mu)
            action = sample.tanh()
            log_probs.append(self.actor.log_prob_from_latent(latent, sample))
            # Target Q heads read the same model states as the online Q heads.
            values.append(torch.minimum(self.critic_target.Q1(latent, action),
                                        self.critic_target.Q2(latent, action)))
            if t == self.imag_horizon:
                break
            actions.append(action)
            pre_tanh.append(sample)
            latent = self.SPR.transition_model.sample_prediction(
                torch.cat([latent, action], dim=-1))
            reward, continuation = self.SPR.predict_outcomes(latent)
            rewards.append(reward)
            continues.append(continuation.sigmoid())
        return dict(latents=torch.stack(latents), actions=torch.stack(actions),
                    pre_tanh=torch.stack(pre_tanh), rewards=torch.stack(rewards),
                    continues=torch.stack(continues), values=torch.stack(values),
                    log_probs=torch.stack(log_probs))

    def update_imagination(self, obs, L, step, scale=1.0):
        if not self.imag_horizon or scale <= 0:
            return
        rollout = self.imagine(obs)
        with torch.no_grad():
            targets = imagined_lambda_return(
                rollout['rewards'], rollout['continues'], rollout['values'],
                rollout['log_probs'], self.discount, self.imag_lambda,
                self.alpha.detach())
            weights = imagined_reach_weights(rollout['continues'], self.discount)
            quantiles = torch.quantile(targets.flatten(), targets.new_tensor([0.05, 0.95]))
            if self.return_quantiles is None:
                self.return_quantiles = quantiles
            else:
                self.return_quantiles.lerp_(quantiles, 0.01)
            normalization = (self.return_quantiles[1] - self.return_quantiles[0]).clamp_min(1)
            latent = rollout['latents'][:-1].flatten(0, 1)
            # The score-function baseline must be independent of the rollout
            # action. Reusing Q(s, a_rollout) would cancel the policy signal.
            _, baseline_action, _, _ = self.actor.forward_latent(latent)
            baseline = torch.minimum(self.critic_target.Q1(latent, baseline_action),
                                     self.critic_target.Q2(latent, baseline_action))
            advantages = (targets - baseline.reshape_as(targets)) / normalization

        latent = rollout['latents'][:-1].flatten(0, 1)
        actions = rollout['actions'].flatten(0, 1)
        weight = weights.flatten(0, 1)
        target = targets.flatten(0, 1)
        q1, q2 = self.critic.Q1(latent, actions), self.critic.Q2(latent, actions)
        critic_loss = (weight * ((q1 - target).square() + (q2 - target).square())).mean()
        if self.imag_critic_weight:
            self.critic_optimizer.zero_grad(set_to_none=True)
            (scale * self.imag_critic_weight * critic_loss).backward()
            self.critic_optimizer.step()

        log_prob = self.actor.log_prob_from_latent(
            latent, rollout['pre_tanh'].flatten(0, 1))
        # A separate reparameterized sample estimates the squashed policy's
        # entropy gradient; REINFORCE above treats rollout actions as constants.
        _, _, entropy_log_prob, _ = self.actor.forward_latent(latent)
        actor_loss = (weight * (-log_prob * advantages.flatten(0, 1) +
                               self.alpha.detach() * entropy_log_prob)).mean()
        if self.imag_actor_weight:
            self.actor_optimizer.zero_grad(set_to_none=True)
            (scale * self.imag_actor_weight * actor_loss).backward()
            self.actor_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/imag_actor_loss', actor_loss, step)
            L.log('train/imag_critic_loss', critic_loss, step)
            L.log('train/imag_return', targets.mean(), step)
            L.log('train/imag_continue', rollout['continues'].mean(), step)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, spr_kwargs = replay_buffer.sample_spr(
            )
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio(
            )

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        if self.encoder_type == 'pixel':
            if spr_kwargs is not None:
                self.update_spr(spr_kwargs, L, step)
                self.model_updates += 1
            if self.reward_loss_weight or self.continue_loss_weight:
                self.update_world_heads(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if self.encoder_type == 'pixel':
            scale = self.imagination_scale
            if step % self.log_interval == 0:
                L.log('train/imag_scale', scale, step)
                L.log('train/model_updates', self.model_updates, step)
            self.update_imagination(obs, L, step, scale)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.encoder,
                                     self.critic_target.encoder,
                                     self.encoder_tau)
            if self.encoder_type == 'pixel':
                utils.soft_update_params(self.SPR.encoder, self.SPR.target_encoder,
                                         self.SPR.momentum_tau)
                utils.soft_update_params(self.SPR.global_classifier,
                                         self.SPR.global_target_classifier,
                                         self.SPR.momentum_tau)

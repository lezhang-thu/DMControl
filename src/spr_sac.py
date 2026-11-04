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


def lambda_returns(rewards, values, discount, lam):
    """Lambda returns for imagined states 0..H-1.

    rewards: [H, B, 1] reward of the transition into the next state.
    values: [H + 1, B, 1] soft values V(z_t) = min Q_target - alpha log pi.
    Continuation is 1, so the only discount is `discount`.
    """
    live = rewards.new_tensor(float(discount))
    mixed = rewards + live * (1.0 - lam) * values[1:]
    returns = values[-1]
    outputs = []
    for index in range(rewards.shape[0] - 1, -1, -1):
        returns = mixed[index] + live * lam * returns
        outputs.append(returns)
    return torch.stack(outputs[::-1], 0)


class Actor(nn.Module):
    """MLP actor. The shared encoder and SPR projection live outside this module."""
    def __init__(self, action_shape, hidden_dim, feature_dim, log_std_min,
                 log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]))

        self.apply(weight_init)

    def forward(self, feature, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.trunk(feature).chunk(2, dim=-1)

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
            entropy = None

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

    def forward(self, obs, action, detach_encoder=False, projection=None):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)
        if projection is not None:
            obs = projection(obs)

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

        self.reward_head = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 1))
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)

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
            auxiliary_task_lr=1e-3,
            imag_warmup=2000,
            imag_actor_weight=0.1,
            imag_value_weight=0.05,
            imag_lambda=0.95,
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
        self.imag_warmup = int(imag_warmup)
        self.imag_actor_weight = imag_actor_weight
        self.imag_value_weight = imag_value_weight
        self.imag_lambda = imag_lambda
        self.grad_steps = 0

        self.critic = Critic(obs_shape, action_shape, hidden_dim, encoder_type,
                             encoder_feature_dim, num_layers,
                             num_filters).to(device)

        self.critic_target = Critic(obs_shape, action_shape, hidden_dim,
                                    encoder_type, encoder_feature_dim,
                                    num_layers, num_filters).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(action_shape, hidden_dim,
                           self.critic.encoder.feature_dim, actor_log_std_min,
                           actor_log_std_max).to(device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        critic_parameters = list(self.critic.parameters())
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
            # The real critic loss trains the shared projection. The predictor
            # and the transition stay on the SPR optimizer only.
            critic_parameters += list(self.SPR.global_classifier.parameters())
            # optimizer for critic encoder and spr loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=2 * encoder_lr)
            self.spr_optimizer = torch.optim.Adam(self.SPR.parameters(),
                                                  lr=auxiliary_task_lr)

        # The actor trunk is the only module the actor loss steps.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=(actor_beta, 0.999))

        self.critic_optimizer = torch.optim.Adam(critic_parameters,
                                                 lr=critic_lr,
                                                 betas=(critic_beta, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=(alpha_beta, 0.999))

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

    def _project(self, latent, target=False):
        if self.encoder_type != 'pixel':
            return latent
        projection = (self.SPR.global_target_classifier
                      if target else self.SPR.global_classifier)
        return projection(latent)

    def _policy(self, obs, compute_pi=True, compute_log_pi=True):
        latent = self._project(self.critic.encoder(obs))
        return self.actor(latent,
                          compute_pi=compute_pi,
                          compute_log_pi=compute_log_pi)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self._policy(obs,
                                       compute_pi=False,
                                       compute_log_pi=False)
            return mu.detach().cpu().numpy().flatten()

    def sample_action(self, obs):
        if self.encoder_type == 'pixel' and obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self._policy(obs, compute_log_pi=False)
            return pi.detach().cpu().numpy().flatten()

    def _imag_scale(self):
        warmup = self.imag_warmup
        if warmup <= 0:
            return 1.0
        if self.grad_steps <= warmup:
            return 0.0
        return float(min(1.0, (self.grad_steps - warmup) / float(warmup)))

    def _note_imagination_schedule(self, logger, step):
        scale = self._imag_scale()
        if self.imag_warmup <= 0 and self.grad_steps == 1:
            print("imaginary training starts at gradient step 1", flush=True)
            print("imaginary training at full strength at gradient step 1",
                  flush=True)
        elif self.grad_steps == self.imag_warmup + 1:
            print("imaginary training starts at gradient step "
                  f"{self.grad_steps}",
                  flush=True)
        elif self.grad_steps == 2 * self.imag_warmup:
            print("imaginary training at full strength at gradient step "
                  f"{self.grad_steps}",
                  flush=True)
        if step % self.log_interval == 0:
            logger.log('train/imag_scale', scale, step)
        return scale

    def _imagination_losses(self, obs, with_actor):
        """On-policy SAC losses inside a short latent rollout.

        Every latent is detached, so these losses do not train the encoder
        or the transition. The reward readout is detached as well.
        """
        horizon = self.jumps
        latent = self.critic.encoder(obs).detach()
        rewards = []
        values = []
        online_q1 = []
        online_q2 = []
        actor_terms = []
        for _ in range(horizon):
            feature = self._project(latent)
            _, action, log_prob, _ = self.actor(feature.detach())
            nxt = self.SPR.transition_model.sample_prediction(
                torch.cat([latent, action.detach()], dim=1)).detach()
            rewards.append(self.SPR.reward_head(nxt).detach())
            with torch.no_grad():
                target_feature = self._project(latent, target=True)
                target_q1 = self.critic_target.Q1(target_feature, action)
                target_q2 = self.critic_target.Q2(target_feature, action)
                values.append(
                    torch.min(target_q1, target_q2) -
                    self.alpha.detach() * log_prob)
            online_q1.append(self.critic.Q1(feature, action.detach()))
            online_q2.append(self.critic.Q2(feature, action.detach()))
            if with_actor:
                actor_q1 = self.critic.Q1(feature.detach(), action)
                actor_q2 = self.critic.Q2(feature.detach(), action)
                actor_terms.append(self.alpha.detach() * log_prob -
                                   torch.min(actor_q1, actor_q2))
            latent = nxt
        with torch.no_grad():
            _, final_action, final_log_prob, _ = self.actor(
                self._project(latent).detach())
            target_feature = self._project(latent, target=True)
            target_q1 = self.critic_target.Q1(target_feature, final_action)
            target_q2 = self.critic_target.Q2(target_feature, final_action)
            values.append(
                torch.min(target_q1, target_q2) -
                self.alpha.detach() * final_log_prob)
        rewards = torch.stack(rewards, 0)
        values = torch.stack(values, 0)
        returns = lambda_returns(rewards, values, self.discount,
                                 self.imag_lambda).detach()
        weights = self.discount**torch.arange(
            horizon, device=obs.device, dtype=rewards.dtype)
        q1 = torch.stack(online_q1, 0)
        q2 = torch.stack(online_q2, 0)
        value_error = (q1 - returns).pow(2) + (q2 - returns).pow(2)
        value_loss = (weights[:, None, None] * value_error).mean()
        if not with_actor:
            return None, value_loss
        actor_error = torch.stack(actor_terms, 0)
        actor_loss = (weights[:, None, None] * actor_error).mean()
        return actor_loss, value_loss

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step,
                      imag_value_loss=None, imag_scale=0.0):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self._policy(next_obs)
            projection = (self.SPR.global_target_classifier
                          if self.encoder_type == 'pixel' else None)
            target_Q1, target_Q2 = self.critic_target(
                next_obs, policy_action, projection=projection)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # Replay action, projected shared encoder feature.
        projection = (self.SPR.global_classifier
                      if self.encoder_type == 'pixel' else None)
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder,
            projection=projection)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        if imag_value_loss is not None and imag_scale > 0.0:
            critic_loss = critic_loss + (
                self.imag_value_weight * imag_scale * imag_value_loss)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update_actor_and_alpha(self, obs, L, step, imag_actor_loss=None,
                               imag_scale=0.0):
        # Detach the projected feature so the actor loss trains the trunk only.
        feature = self._project(self.critic.encoder(obs)).detach()
        _, pi, log_pi, log_std = self.actor(feature)
        actor_Q1 = self.critic.Q1(feature, pi)
        actor_Q2 = self.critic.Q2(feature, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        if imag_actor_loss is not None and imag_scale > 0.0:
            actor_loss = actor_loss + (
                self.imag_actor_weight * imag_scale * imag_actor_loss)

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
        # [1+self.jumps, 1]
        reward = spr_kwargs["reward"]

        pred_latents = []

        input_obs = observation[0].flatten(1, 2)
        input_obs = self.SPR.transform(input_obs, augment=True)

        # stem forward
        lead_dim, T, B, img_shape = utils.infer_leading_dims(input_obs, 3)
        latent = self.SPR.encoder(input_obs.reshape(
            T * B, *img_shape))  # Fold if T dimension.

        # Open-loop rollout with the replay actions.
        pred_latents.append(latent)
        for j in range(1, self.jumps + 1):
            latent = self.SPR.transition_model.sample_prediction(
                torch.cat([latent, action[j - 1]], dim=1))
            pred_latents.append(latent)

        reward_loss = self._arrival_reward_loss(pred_latents, observation,
                                                reward)
        spr_loss = self.SPR.do_spr_loss(pred_latents, observation)
        if self.jumps > 0:
            model_spr_loss = spr_loss[1:].mean(0)  # (jumps, bs) => (bs, )
            t0_spr_loss = spr_loss[0]
        else:
            t0_spr_loss = spr_loss[0]
            model_spr_loss = torch.zeros_like(spr_loss)

        final_spr_loss = self.t0_spr_loss_weight * t0_spr_loss.mean() + \
            self.model_spr_weight * model_spr_loss.mean() + \
            self.reward_loss_weight * reward_loss

        self.encoder_optimizer.zero_grad()
        self.spr_optimizer.zero_grad()
        final_spr_loss.backward()
        self.encoder_optimizer.step()
        self.spr_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/spr_loss', final_spr_loss, step)
            L.log('train/reward_loss', reward_loss, step)

    def _arrival_reward_loss(self, pred_latents, observation, reward):
        """Detached readout: the latent of s_{k+1} predicts reward r_k."""
        if self.jumps <= 0 or self.reward_loss_weight == 0.0:
            return pred_latents[0].new_zeros(())
        rollout = torch.stack(pred_latents[1:], 0).detach()
        predicted_rollout = self.SPR.reward_head(
            rollout.reshape(-1, rollout.shape[-1])).view(self.jumps, -1, 1)
        real = self._real_arrival_features(observation)
        predicted_real = self.SPR.reward_head(
            real.reshape(-1, real.shape[-1])).view(self.jumps, -1, 1)
        target = reward[:self.jumps]
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        return (F.mse_loss(predicted_rollout, target) +
                F.mse_loss(predicted_real, target))

    def _real_arrival_features(self, observation):
        frames = observation[1:self.jumps + 1]
        frames = frames.transpose(0, 1).flatten(2, 3)
        frames = self.SPR.transform(frames, True)
        with torch.no_grad():
            latents = self.SPR.target_encoder(frames.flatten(0, 1))
        batch = observation.shape[1]
        return latents.view(batch, self.jumps, -1).transpose(0, 1)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, spr_kwargs = replay_buffer.sample_spr(
            )
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio(
            )

        self.grad_steps += 1
        imag_scale = self._note_imagination_schedule(L, step)
        imagine = (imag_scale > 0.0 and self.encoder_type == 'pixel' and
                   self.jumps > 0)
        imag_value_loss = None
        if imagine:
            _, imag_value_loss = self._imagination_losses(obs, with_actor=False)

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)
        self.update_critic(obs, action, reward, next_obs, not_done, L, step,
                           imag_value_loss=imag_value_loss,
                           imag_scale=imag_scale)
        if self.encoder_type == 'pixel':
            self.update_spr(spr_kwargs, L, step)

        if step % self.actor_update_freq == 0:
            # A fresh rollout. The critic step above updates Q in place, so
            # the actor graph must be built after that step.
            imag_actor_loss = None
            if imagine:
                imag_actor_loss, _ = self._imagination_losses(
                    obs, with_actor=True)
            self.update_actor_and_alpha(obs, L, step,
                                        imag_actor_loss=imag_actor_loss,
                                        imag_scale=imag_scale)

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

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


class Actor(nn.Module):
    """MLP actor network. The shared encoder and policy projection live outside."""
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


def make_spr_projection(feature_dim, latent_dim):
    """Projection used by one SPR branch. Same width as the encoder feature."""
    return nn.Sequential(nn.Linear(feature_dim, latent_dim), nn.ReLU(),
                         nn.Linear(latent_dim, feature_dim))


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
        self.q_proj = make_spr_projection(encoder_feature_dim, latent_dim)
        self.q_predictor = make_spr_projection(encoder_feature_dim, latent_dim)
        self.q_proj_target = make_spr_projection(encoder_feature_dim, latent_dim)
        self.pi_proj = make_spr_projection(encoder_feature_dim, latent_dim)
        self.pi_predictor = make_spr_projection(encoder_feature_dim, latent_dim)
        self.pi_proj_target = make_spr_projection(encoder_feature_dim, latent_dim)
        self.q_proj_target.load_state_dict(self.q_proj.state_dict())
        self.pi_proj_target.load_state_dict(self.pi_proj.state_dict())
        for target in (self.q_proj_target, self.pi_proj_target):
            for parameter in target.parameters():
                parameter.requires_grad = False
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

    def branch_spr_loss(self, predictions, targets, observation):
        """Normalized squared error of one branch, shaped (time, batch)."""
        predictions = predictions.reshape(-1, observation.shape[1], self.jumps + 1,
                                          predictions.shape[-1]).transpose(1, 2)
        targets = targets.reshape(-1, observation.shape[1], self.jumps + 1,
                                  targets.shape[-1]).transpose(1, 2)
        return self.spr_loss(predictions, targets)

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
            q_targets = self.q_proj_target(target_latents)
            pi_targets = self.pi_proj_target(target_latents)

        q_predictions = self.q_predictor(self.q_proj(latents))
        pi_predictions = self.pi_predictor(self.pi_proj(latents))
        # Each branch is normalized on its own. The two losses have equal weight.
        spr_loss = (self.branch_spr_loss(q_predictions, q_targets, observation) +
                    self.branch_spr_loss(pi_predictions, pi_targets, observation))
        # split to batch, jumps
        spr_loss = spr_loss.reshape(-1, observation.shape[1])
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
            reward_loss_weight=0.0,
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

        actor_parameters = list(self.actor.parameters())
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
            # Actor loss trains the policy projection and the shared encoder.
            # Critic loss trains the Q projection. Predictors stay on the SPR step.
            actor_parameters += list(self.SPR.pi_proj.parameters())
            actor_parameters += list(self.critic.encoder.parameters())
            critic_parameters += list(self.SPR.q_proj.parameters())
            # The shared encoder gets one SPR step, through this optimizer only.
            encoder_parameters = list(self.critic.encoder.parameters())
            self.encoder_optimizer = torch.optim.Adam(
                encoder_parameters, lr=2 * encoder_lr)
            encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
            spr_parameters = [
                parameter for parameter in self.SPR.parameters()
                if parameter.requires_grad and id(parameter) not in encoder_parameter_ids
            ]
            self.spr_optimizer = torch.optim.Adam(spr_parameters,
                                                  lr=auxiliary_task_lr)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(actor_parameters,
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

    def _policy(self, obs, compute_pi=True, compute_log_pi=True):
        """Actor trunk on pi_proj(z). Samples a fresh action when requested."""
        z = self.critic.encoder(obs)
        if self.encoder_type == 'pixel':
            z = self.SPR.pi_proj(z)
        return self.actor(z, compute_pi=compute_pi, compute_log_pi=compute_log_pi)

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

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            # Fresh policy sample for the bootstrap. The replay action below
            # is the one stored with this transition. Neither is a world-model input.
            _, policy_action, log_pi, _ = self._policy(next_obs)
            q_projection = (self.SPR.q_proj_target
                            if self.encoder_type == 'pixel' else None)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action,
                                                      projection=q_projection)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates from the replay action
        q_projection = self.SPR.q_proj if self.encoder_type == 'pixel' else None
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder,
            projection=q_projection)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update_actor_and_alpha(self, obs, L, step):
        # Fresh sample from the current policy. Gradient enters the shared
        # encoder through pi_proj. The Q state feature is stopped so this loss
        # does not train q_proj; the sample itself still trains the trunk.
        z = self.critic.encoder(obs)
        if self.encoder_type == 'pixel':
            policy_feature = self.SPR.pi_proj(z)
            value_feature = self.SPR.q_proj(z).detach()
        else:
            policy_feature = z
            value_feature = z.detach()
        _, pi, log_pi, log_std = self.actor(policy_feature)
        actor_Q1 = self.critic.Q1(value_feature, pi)
        actor_Q2 = self.critic.Q2(value_feature, pi)

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
        # [1+self.jumps, 1]
        reward = spr_kwargs["reward"]

        pred_reward = []
        pred_latents = []

        input_obs = observation[0].flatten(1, 2)
        input_obs = self.SPR.transform(input_obs, augment=True)

        # stem forward
        lead_dim, T, B, img_shape = utils.infer_leading_dims(input_obs, 3)
        latent = self.SPR.encoder(input_obs.reshape(
            T * B, *img_shape))  # Fold if T dimension.

        # spr loss
        pred_latents.append(latent)
        if self.jumps > 0:
            pred_rew = self.SPR.reward_predictor(pred_latents[0])
            pred_reward.append(F.log_softmax(pred_rew, -1))

            for j in range(1, self.jumps + 1):
                latent = self.SPR.transition_model.sample_prediction(
                    torch.cat([latent, action[j - 1]], dim=1))
                pred_rew = self.SPR.reward_predictor(
                    latent)[:observation.shape[1]]
                pred_latents.append(latent)
                pred_reward.append(F.log_softmax(pred_rew, -1))

        # pred_reward loss
        if len(pred_reward) > 0:
            pred_reward = torch.stack(pred_reward, 0)
            with torch.no_grad():
                reward_target = utils.to_categorical(
                    reward[:self.jumps + 1].flatten().to(self.device),
                    limit=0).reshape(*pred_reward.shape)
            reward_loss = -torch.sum(reward_target * pred_reward,
                                     2).mean(0).cpu()
        else:
            reward_loss = torch.zeros(observation.shape[1], )

        spr_loss = self.SPR.do_spr_loss(pred_latents, observation)
        if self.jumps > 0:
            model_spr_loss = spr_loss[1:].mean(0)  # (jumps, bs) => (bs, )
            t0_spr_loss = spr_loss[0]
        else:
            t0_spr_loss = spr_loss[0]
            model_spr_loss = torch.zeros_like(spr_loss)

        final_spr_loss = self.t0_spr_loss_weight * t0_spr_loss.mean() + \
            self.model_spr_weight * model_spr_loss.mean() + \
            self.reward_loss_weight * reward_loss.mean().to(self.device)

        self.encoder_optimizer.zero_grad()
        self.spr_optimizer.zero_grad()
        final_spr_loss.backward()
        self.encoder_optimizer.step()
        self.spr_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/spr_loss', final_spr_loss, step)

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
            if self.encoder_type == 'pixel':
                utils.soft_update_params(self.SPR.encoder, self.SPR.target_encoder,
                                         self.SPR.momentum_tau)
                utils.soft_update_params(self.SPR.q_proj, self.SPR.q_proj_target,
                                         self.SPR.momentum_tau)
                utils.soft_update_params(self.SPR.pi_proj, self.SPR.pi_proj_target,
                                         self.SPR.momentum_tau)

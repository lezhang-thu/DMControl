# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import time

from env import make as make_env
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import utils
from logger import Logger
from spr_sac import SPRSacAgent


def evaluate(env, agent, num_episodes, L, step, args):
    rewards = []
    start_time = time.time()
    with utils.eval_mode(agent):
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                obs = utils.center_crop_image(obs, args.image_size)
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            L.log('eval/episode_reward', episode_reward, step)
            rewards.append(episode_reward)

    L.log('eval/eval_time', time.time() - start_time, step)
    L.log('eval/mean_episode_reward', np.mean(rewards), step)
    L.log('eval/best_episode_reward', np.max(rewards), step)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    return SPRSacAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        transition_model_layer_width=args.transition_model_layer_width,
        jumps=args.jumps,
        latent_dim=args.latent_dim,
        momentum_tau=args.momentum_tau,
        model_spr_weight=args.fp_loss_weight,
        auxiliary_task_lr=args.auxiliary_task_lr,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        log_interval=args.log_interval)


def make_environment(args):
    env = make_env(domain_name=args.domain_name,
                   task_name=args.task_name,
                   seed=args.seed,
                   height=args.pre_transform_image_size,
                   width=args.pre_transform_image_size,
                   frame_skip=args.action_repeat)
    return utils.FrameStack(env, k=args.frame_stack)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    args.init_steps *= args.action_repeat
    args.log_interval *= args.action_repeat
    args.actor_update_freq *= args.action_repeat
    args.critic_target_update_freq *= args.action_repeat
    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
    utils.set_seed_everywhere(args.seed)
    env = make_environment(args)
    eval_env = make_environment(args)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m%dT%H", ts)
    exp_name = 'ETA-w2'
    env_name = exp_name + '-' + args.domain_name + '-' + args.task_name
    exp_name = env_name + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed) + '-spr_sac'

    exp_name = exp_name + '-k' + str(args.jumps)
    args.work_dir = args.work_dir + '/' + 'spr_sac_dmc' + '/' + exp_name
    os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    pre_aug_obs_shape = env.observation_space.shape
    obs_shape = (pre_aug_obs_shape[0], args.image_size, args.image_size)

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        auxiliary_task_batch_size=args.auxiliary_task_batch_size,
        jumps=args.jumps,
    )

    agent = make_agent(obs_shape=obs_shape,
                       action_shape=action_shape,
                       args=args,
                       device=device)

    L = Logger(args.work_dir)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(0, args.num_env_steps, args.action_repeat):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(eval_env, agent, args.num_eval_episodes, L, step, args)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            agent.update(replay_buffer, L, step)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs

    L.dump(args.num_env_steps)
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()

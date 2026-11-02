# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

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
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for _ in range(num_episodes):
            obs, _ = env.reset()

            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent != 'spr_sac':
        raise ValueError(f'Only spr_sac is supported, got {args.agent!r}')
    return SPRSacAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        augmentation=args.augmentation,
        transition_model_type=args.transition_model_type,
        transition_model_layer_width=args.transition_model_layer_width,
        jumps=args.jumps,
        latent_dim=args.latent_dim,
        time_offset=args.time_offset,
        momentum_tau=args.momentum_tau,
        aug_prob=args.aug_prob,
        model_spr_weight=args.fp_loss_weight,
        reward_loss_weight=args.reward_loss_weight,
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
        encoder_type=args.encoder_type,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        log_interval=args.log_interval,
        detach_encoder=args.detach_encoder)


def make_environment(args):
    env = make_env(domain_name=args.domain_name,
                       task_name=args.task_name,
                       seed=args.seed,
                       visualize_reward=False,
                       from_pixels=args.encoder_type == 'pixel',
                       height=args.pre_transform_image_size,
                       width=args.pre_transform_image_size,
                       frame_skip=args.action_repeat)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    return env


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    args.init_steps *= args.action_repeat
    args.log_interval *= args.action_repeat
    args.actor_update_freq *= args.action_repeat
    args.critic_target_update_freq *= args.action_repeat
    if args.seed is None:
        args.seed = args.seed_and_gpuid[0]
    if args.gpuid is None:
        args.gpuid = args.seed_and_gpuid[1]
    args.domain_name = args.domain_name or args.env_name.split('/')[0]
    args.task_name = args.task_name or args.env_name.split('/')[1]
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
    # exp_name = 'aug_then_expand_AN10'    # re-run
    exp_name = 'ETA-w2'    # re-run
    env_name = exp_name + '-' + args.domain_name + '-' + args.task_name
    exp_name = env_name + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed) + '-' + args.agent

    exp_name = exp_name + '-k' + str(args.jumps)
    args.work_dir = args.work_dir + '/' + f'{args.agent}_dmc' + '/' + exp_name
    utils.make_dir(args.work_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack,
                             args.pre_transform_image_size,
                             args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

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
    replay_buffer.add_agent(agent)

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
            num_updates = 1
            for _ in range(num_updates):
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
    torch.multiprocessing.set_start_method('spawn')
    main()

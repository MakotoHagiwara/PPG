from gym.wrappers import Monitor

import matplotlib
import numpy as np
import gym
import torch
from PPG_Agent import PpgAgent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pyglet

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import datetime

gym.logger.set_level(40)
from collections import deque
from GymWrapper import ImgWrapper
from util import layer_init
from Noise import ActionNoise

env = None
is_image = False
if is_image:
    env = ImgWrapper(gym.make('LunarLanderContinuous-v2'), gray_scale=True)
else:
    env = gym.make('LunarLanderContinuous-v2')

from mpi4py import MPI

# setup mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# random seed or fiexd seed
fixed_seed = False
if fixed_seed:
    seed = rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    env.seed(seed)
else:
    seed = 'random_{}'.format(rank)

max_training_steps = int(1e6)  # max steps for training

# hparams
num_updates = 32  # aux task interval against PPO update
aux_num_updates = 6  # epoch of aux task
aux_critic_loss_coef = float(0.01)   # reduce critic loss in auxialry_phase by multiplying coef
beta_clone = 1.0  # bc loss coef
rollout_length = 8192  # change fron 2048

memory_size = rollout_length * num_updates + 100  # tekito

clip_aux_critic_loss = None  # clipping, set float value to clip loss
clip_aux_multinet_critic_loss = None  # clipping, set float value to clip loss
multipleet_upadte_clip_grad_norm = None  # gradient clipping
# ログ用の設定
episode_rewards = []
num_average_epidodes = 25  # moving average
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# set this flag True to change spearetd TPO (no auxialry update)
debug_no_aux_phase = False
algo = 'PPO' if debug_no_aux_phase else 'PPG'

logdir = './final_result/{}/num_updates-{}_aux_num_updats-{}_beta_clone{}'\
         '_aux_critic_coef-{}_clip_aux_multinet_critic_loss-{}_'\
         'clip_aux_critic_loss-{}_multipleet_upadte_clip_grad_norm-'\
         '{}_rollout_lenght-{}/seed_{}'.format(algo,
                                               num_updates,
                                               aux_num_updates,
                                               beta_clone,
                                               aux_critic_loss_coef,
                                               clip_aux_multinet_critic_loss,
                                               clip_aux_critic_loss,
                                               multipleet_upadte_clip_grad_norm,
                                               rollout_length,
                                               seed)
writer = SummaryWriter(log_dir=logdir)
    

print('loging to {}...'.format(logdir))
max_steps = env.spec.max_episode_steps  # エピソードの最大ステップ数

# create agent
agent = PpgAgent(env.observation_space,
                 env.action_space,
                 device,
                 memory_size=memory_size,
                 writer=writer,
                 is_image = is_image,
                 rollout_length=rollout_length,
                 debug_no_aux_phase=debug_no_aux_phase,
                 num_updates=num_updates,
                 aux_num_updates=aux_num_updates,
                 aux_critic_loss_coef=aux_critic_loss_coef,
                 clip_aux_critic_loss=clip_aux_critic_loss,
                 clip_aux_multinet_critic_loss=clip_aux_multinet_critic_loss,
                 multipleet_upadte_clip_grad_norm=multipleet_upadte_clip_grad_norm,
                 beta_clone=beta_clone)

global_step = 0
eval_cnt = 0
episode = 0
eval_interval = 0.01e6
while global_step < max_training_steps:
    state = env.reset()
    episode_reward = 0
    noise = ActionNoise(env.action_space.shape[0])
    for t in range(max_steps):
        action, log_pis = agent.get_action(state)  #  行動を選択o
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        done_masked = False if t == (max_steps - 1) else done
        agent.memory.add(state, action, reward, done_masked, log_pis)
        agent.update(next_state)  # actorとcriticを更新
        state = next_state
        global_step += 1
        if done:
            break
    episode += 1
    episode_rewards.append(episode_reward)
    writer.add_scalar("reward", episode_reward, global_step)

    #  evaluation in every 0.1M steps
    if global_step > (eval_cnt * eval_interval):
        sum_reward = 0.0
        agent.memory.save_memory()
        for k in range(50):
            state = env.reset()
            done = False
            step = 0
            while not done and step < max_steps:
                step += 1
                action, log_pis = agent.get_action(state)  #  行動を選択
                next_state, reward, done, _ = env.step(action)
                sum_reward += reward
                state = next_state
        writer.add_scalar("eval_reward", sum_reward / 50, eval_cnt * eval_interval)
        print("Experiment num: %d evaluation: steps %d episode %d finished | Average reward %f" % (rank, global_step, episode, sum_reward / 50))
        eval_cnt += 1


# 累積報酬の移動平均を表示
moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)), moving_average)
plt.title('PPG: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

env.close()

# from pyvirtualdisplay import Display
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
#検証用にシードを固定する
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
env.seed(seed)

max_training_steps = int(0.5e6)  # 0.5M steps training
memory_size = 256 * 32 + 1

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
#検証用にシードを固定する
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
env.seed(seed)

max_training_steps = int(1e6)  # 1M steps training
memory_size = 2048 * 32 + 100

# ログ用の設定
episode_rewards = []
num_average_epidodes = 25  # moving average
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set this flag True to change spearetd TPO (no auxialry update)
debug_no_aux_phase = False
prefix = 'TPO' if debug_no_aux_phase else 'PPG'

logdir = './results/{}/seed_{}'.format(prefix, seed)
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
                 debug_no_aux_phase=debug_no_aux_phase)

global_step = 0
eval_cnt = 0
episode = 0
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
    if global_step > (eval_cnt * int(0.1e6)):
        eval_cnt += 1
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
        writer.add_scalar("eval_reward", sum_reward / 50, global_step)
        print("Evaluation: steps %d episode %d finished | Average reward %f" % (global_step, episode, sum_reward / 50))

# 累積報酬の移動平均を表示
moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)), moving_average)
plt.title('PPG: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

env.close()


# ログ用の設定
episode_rewards = []
num_average_epidodes = 25  # moving average
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set this flag True to change spearetd TPO (no auxialry update)
debug_no_aux_phase = False
prefix = 'TPO' if debug_no_aux_phase else 'PPG'

logdir = './results/{}/seed_{}'.format(prefix, seed)
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
                 debug_no_aux_phase=debug_no_aux_phase)

global_step = 0
eval_cnt = 0
episode = 0
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
    if global_step > (eval_cnt * int(0.1e6)):
        eval_cnt += 1
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
        writer.add_scalar("eval_reward", sum_reward / 50, global_step)
        print("Evaluation: steps %d episode %d finished | Average reward %f" % (global_step, episode, sum_reward / 50))

# 累積報酬の移動平均を表示
moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)), moving_average)
plt.title('PPG: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

env.close()

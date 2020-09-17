import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import gym
import cv2
import math
import torch.nn as nn
from torch.distributions import Normal

def layer_init(layer):
    denominator = math.sqrt(layer.in_features)
    nn.init.uniform_(layer.weight, -1 / denominator, 1 / denominator)

def actor_last_layer_init(layer):
    nn.init.uniform_(layer.weight, -3e-3, 3e-3)


def critic_last_layer_init(layer):
    nn.init.uniform_(layer.weight, -3e-4, 3e-4)
    
def calculate_advantage(values, rewards, dones, gamma = 0.999, lambda_ = 0.997):
    #values[1:]:s_{t+1}の状態価値, values[:-1]:s_tの状態価値
    #valuesは0~t+1の状態が格納されている???
    #rewardsのサイズをnとするとvaluesのサイズはn+1
    deltas = rewards + gamma * values[1:] * (1-dones) - values[:-1]
    
    #rewardsと同じサイズのランダムに初期化された変数を生成
    advantages = torch.empty_like(rewards)
    
    #X[-1]は一番後ろの要素の値をとってくる
    advantages[-1] = deltas[-1]
    
    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = deltas[t] + gamma * lambda_ * (1 - dones[t]) * advantages[t + 1]
    
    #X[:-1]は一番最後以外の要素をとってくる
    targets = advantages + values[:-1]
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return targets, advantages

def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim = -1, keepdim = True)
    log_pis = torch.clamp(log_pis, max = 10, min = -10)
    return log_pis

def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    actions = torch.tanh(means + stds * noises)
    log_pis = calculate_log_pi(log_stds, noises, actions)
    return actions, log_pis

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def evaluate_log_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-6)
    return calculate_log_pi(log_stds, noises, actions)
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import gym
import cv2
import math
import torch.nn as nn

class ImgWrapper(gym.Wrapper):
    def __init__(self, env, gray_scale=False, img_shape=(84, 84)):
        gym.Wrapper.__init__(self, env)

        self.gray_scale = gray_scale
        self.img_shape = img_shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1 if gray_scale else 3,) + img_shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._obs()

    def step(self, action):
        _, rew, done, info = self.env.step(action)
        return self._obs(), rew, done, info

    def _obs(self):
        obs = self.env.render(mode='rgb_array')
        obs = cv2.resize(obs, self.img_shape, interpolation=cv2.INTER_AREA)
        if self.gray_scale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[None, ...]
        return obs.astype(np.uint8).copy()    
        #return np.transpose(obs, axes=(2, 0, 1)).astype(np.uint8).copy()
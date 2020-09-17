import copy
from PPG import CriticNetwork
from PPG import MultipleNetwork
from Memory import ReplayMemory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import math
import util

class PpgAgent:
    def __init__(self, observation_space, action_space, device, gamma = 0.995,
                 actor_lr = 5e-4, critic_lr = 5e-4, batch_size = 128,
                 memory_size = 50000, tau = 5e-3, weight_decay = 1e-2,
                 sigma = 0.2, noise_clip = 0.5,
                 alpha = 0.2, alpha_lr = 3e-4, rollout_length = 2048, lambda_ = 0.95, beta_clone = 1.0,
                 coef_ent = 0.01, num_updates = 32, policy_epoch = 1, value_epoch = 1, aux_num_updates = 6, aux_epoch_batch = 16, max_grad_norm=0.5, clip_eps = 0.2, writer = None, is_image = False):
        super(PpgAgent, self).__init__()
        self.action_mean = (0.5 * (action_space.high + action_space.low))[0]
        self.action_halfwidth = (0.5 * (action_space.high - action_space.low))[0]
        self.num_state = observation_space.shape[0]
        self.num_action = action_space.shape[0]
        self.state_mean = None
        self.state_halfwidth = None
        if abs(observation_space.high[0]) != math.inf:
            self.state_mean = 0.5 * (observation_space.high + observation_space.low)
            self.state_halfwidth = 0.5 * (observation_space.high - observation_space.low)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.multipleNet = MultipleNetwork(self.num_state, action_space, device, is_image = is_image).to(self.device)
        self.multipleNet_target = MultipleNetwork(self.num_state, action_space, device, is_image = is_image).to(self.device)
        self.multipleNet_target.load_state_dict(self.multipleNet.state_dict())
        self.multipleNet_optimizer = optim.Adam(self.multipleNet.parameters(), lr=actor_lr)
        
        self.critic = CriticNetwork(self.num_state, action_space, device, is_image = is_image).to(self.device)
        self.critic_target = CriticNetwork(self.num_state, action_space, device, is_image = is_image).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr, weight_decay=weight_decay)
        
        self.alpha = alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr = alpha_lr)
        
        self.memory = ReplayMemory(observation_space, action_space, device, num_state = self.num_state, memory_size = memory_size, is_image = is_image)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tau = tau
        self.writer = writer
        self.update_step = 0
        self.is_image =is_image
        self.sigma = sigma
        self.noise_clip = noise_clip
        self.rollout_length = rollout_length
        self.lambda_ = lambda_
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.aux_num_updates = aux_num_updates
        self.clip_eps = clip_eps
        self.beta_clone = beta_clone
        self.policy_epoch = policy_epoch
        self.value_epoch = value_epoch
        self.num_updates = num_updates
        self.aux_epoch_batch = aux_epoch_batch
        
    def normalize_state(self, state):
        if self.state_mean is None:
            return state
        state = (state - self.state_mean) / self.state_halfwidth
        return state
    
    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    def is_update(self, steps):
        return steps % self.rollout_length == 0
    
    def update(self, state = None):
        if not self.is_update(self.memory.index):
            return
        self.update_step += 1
        with torch.no_grad():
            batch = self.memory.sample(state)
            #各サンプルにおける状態行動の値を取ってくる
            action_batch = batch['actions'].to(self.device)
            state_batch = batch['obs'].to(self.device)
            reward_batch = batch['rewards'].to(self.device)
            terminate_batch = batch['terminates'].to(self.device)
            log_pis_batch = batch['log_pis'].to(self.device)
            
            values = self.critic(state_batch)
        targets, advantages = util.calculate_advantage(values, reward_batch, terminate_batch, self.gamma, self.lambda_)
        for j in range(self.num_updates):
            for i in range(max(self.policy_epoch, self.value_epoch)):
                indices = np.arange(self.rollout_length)
                np.random.shuffle(indices)
                for start in range(0, self.rollout_length, self.batch_size):
                    idxes = indices[start:start + self.batch_size]
                    if self.policy_epoch > i:
                        self.update_MultipleNet(state_batch[idxes], action_batch[idxes], log_pis_batch[idxes], advantages[idxes])
                    if self.value_epoch > i:
                        self.update_critic(state_batch[idxes], targets[idxes])
        with torch.no_grad():
            log_pis_old = self.multipleNet.evaluate_log_pi(state_batch[:-1], action_batch)
        for _ in range(self.aux_num_updates):
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)
            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start + self.batch_size]
                self.update_actor_Auxiliary(state_batch[idxes], action_batch[idxes], log_pis_old[idxes], targets[idxes], advantages[idxes])
                self.update_critic_Auxiliary(state_batch[idxes], targets[idxes])
        self.multipleNet.eval()
        self.critic.eval()
        
    def update_actor_Auxiliary(self, states, actions, log_pis_old, targets, advantages):
        loss_critic = (self.multipleNet.q_forward(states) - targets).pow_(2).mean()
        log_pis = self.multipleNet.evaluate_log_pi(states, actions)
        ratios = (log_pis - log_pis_old).exp_()
        loss_a1 = -ratios * advantages
        loss_a2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_a = torch.max(loss_a1, loss_a2).mean()
        loss_joint = loss_critic + self.beta_clone * loss_a
        self.multipleNet_optimizer.zero_grad()
        loss_joint.backward(retain_graph=False)
        self.multipleNet_optimizer.step()
        #if self.update_step % 50 == 0:
        #    print("aux actor loss:", loss_joint.item())
        
    def update_critic_Auxiliary(self, states, targets):
        loss_critic_aux = (self.critic(states) - targets).pow_(2).mean()
        self.critic_optimizer.zero_grad()
        loss_critic_aux.backward(retain_graph=False)
        #nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        #if self.update_step % 50 == 0:
        #    print("aux citic loss:", loss_critic_aux.item())
        
    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()
        self.critic_optimizer.zero_grad()
        loss_critic.backward(retain_graph=False)
        #nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        #if self.update_step % 50 == 0:
        #    print("citic loss:", loss_critic.item())

    def update_MultipleNet(self, states, actions, log_pis_old, advantages):
        log_pis = self.multipleNet.evaluate_log_pi(states, actions)
        if self.update_step % 50 == 0:
            print("log_pis:", log_pis)
        mean_ent = - log_pis.mean()
        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_ent
        self.multipleNet_optimizer.zero_grad()
        loss_actor.backward(retain_graph=False)
        #nn.utils.clip_grad_norm_(self.multipleNet.parameters(), self.max_grad_norm)
        self.multipleNet_optimizer.step()
        if self.update_step % 50 == 0:
            print("actor loss:", loss_actor.item())
        
    # Q値が最大の行動を選択
    def get_action(self, state):
        self.multipleNet.eval()
        if not self.is_image:
            state_tensor = torch.tensor(self.normalize_state(state), dtype=torch.float).view(-1, self.num_state).to(self.device)
        else:
            state_tensor = torch.tensor(state.copy() / 255., dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_pis = self.multipleNet.sample(state_tensor)
            action = action.view(self.num_action).to('cpu').detach().numpy().copy()
        return action, log_pis
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class MultipleNetwork(nn.Module):
    def __init__(self, num_state, action_space, device, hidden_size = 200, is_image = False):
        super(MultipleNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5 * (action_space.high + action_space.low), dtype = torch.float).to(device)
        self.action_halfwidth = torch.tensor(0.5 * (action_space.high - action_space.low), dtype = torch.float).to(device)
        
        self.conv1 = nn.Conv2d(num_state, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        linear_input_size = 7 * 7 * 32
        if not is_image:
            linear_input_size = num_state
            input_size = 64
            self.fc = nn.Linear(linear_input_size, input_size)
            util.layer_init(self.fc)
        else:
            input_size = linear_input_size
        self.criticModule = CriticModule(num_state, action_space, device, hidden_size = input_size)
        self.actorModule= ActorModule(num_state, action_space, device, hidden_size = input_size)
        self.is_image = is_image
        self.action_high = 1.5
        self.action_low = -1.5
        
    def forward(self, state):
        if self.is_image:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
        else:
            x = F.relu(self.fc(state))
        mean, std = self.actorModule.forward(x)
        q = self.criticModule.forward(x)
        std = torch.clamp(std, min = LOG_SIG_MIN, max = LOG_SIG_MAX)
        mean = torch.clamp(mean, min = self.action_low, max = self.action_high)
        return mean, std, q
    
    def q_forward(self, state):
        if self.is_image:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
        else:
            x = F.relu(self.fc(state))
        q = self.criticModule.forward(x)
        return q
    
    def sample(self, state):
        mean, log_stds, q = self.forward(state)
        action, log_pis = util.reparameterize(mean, log_stds)
        action = action * self.action_halfwidth + self.action_mean
        return action, log_pis
    
    def evaluate_log_pi(self, state, actions):
        means, log_stds, q = self.forward(state)
        actions = (actions - self.action_mean) / self.action_halfwidth
        return util.evaluate_log_pi(means, log_stds, actions)

class ActorModule(nn.Module):
    def __init__(self, num_state, action_space, device, hidden_size = 200, is_image = False):
        super(ActorModule, self).__init__()
        self.action_mean = torch.tensor(0.5 * (action_space.high + action_space.low), dtype = torch.float).to(device)
        self.action_halfwidth = torch.tensor(0.5 * (action_space.high - action_space.low), dtype = torch.float).to(device)
        self.fc_last_mean = nn.Linear(hidden_size, action_space.shape[0])
        self.fc_last_std = nn.Linear(hidden_size, action_space.shape[0])
        self.is_image = is_image
        util.actor_last_layer_init(self.fc_last_mean)
        util.actor_last_layer_init(self.fc_last_std)
        
    def forward(self, x):
        mean = self.fc_last_mean(x)
        std = self.fc_last_std(x)
        return mean, std

class CriticModule(nn.Module):
    def __init__(self, num_state, action_space, device, hidden_size = 64, is_image = False):
        super(CriticModule, self).__init__()
        self.action_mean = torch.tensor(0.5 * (action_space.high + action_space.low), dtype = torch.float).to(device)
        self.action_halfwidth = torch.tensor(0.5 * (action_space.high - action_space.low), dtype = torch.float).to(device)
        self.fc_last = nn.Linear(hidden_size, 1)
        self.is_image = is_image
        util.critic_last_layer_init(self.fc_last)
        
    def forward(self, x):
        q = self.fc_last(x)
        return q

class CriticNetwork(nn.Module):
    def __init__(self, num_state, action_space, device, is_image = False):
        super(CriticNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5 * (action_space.high + action_space.low), dtype = torch.float).to(device)
        self.action_halfwidth = torch.tensor(0.5 * (action_space.high - action_space.low), dtype = torch.float).to(device)
        self.conv1 = nn.Conv2d(num_state, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        linear_input_size = 7 * 7 * 32
        if not is_image:
            linear_input_size = num_state
            self.fc1 = nn.Linear(linear_input_size, 64)
            self.fc2 = nn.Linear(64, 64)
            util.layer_init(self.fc1)
            util.layer_init(self.fc2)
            self.fc_last = nn.Linear(64, 1)
        else:
            self.fc1 = nn.Linear(linear_input_size, 64)
            util.layer_init(self.fc1)
            self.fc_last = nn.Linear(64, 1)
        self.is_image = is_image
        util.critic_last_layer_init(self.fc_last)
        
    def forward(self, state):
        if self.is_image:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
        q = self.fc_last(x)
        return q
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math 
from linguamechanica.dql.replay_memory import ReplayBuffer

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, state_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.var = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu  = torch.arctan(self.mu(x)) * 0.1
        var = 1e-5 + (F.relu6(self.var(x)) / 6.0) * 0.1
        v   = self.v(x)
        return (mu, var, v)

class Agent():
    def __init__(self, lr, state_dims,  n_actions,  fc1_dims=256, fc2_dims=256,
                 gamma=0.99, mem_size=50000):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, state_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        self.log_prob = None
        self.memory = ReplayBuffer(mem_size, state_dims, n_actions)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def compute_log_prob(self, mu_v, var_v, actions_v):
        log_prob_part_1 = ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
        log_prob_part_2 = torch.log(torch.sqrt(2 * math.pi * var_v))
        #NOTE: It is addition as it is a multiplication in the non-log domain,
        # but in the log space it is a sum. There is a single probability.
        log_prob = - (log_prob_part_1 + log_prob_part_2)
        return log_prob

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor_critic.device)
        mu_v, var_v, _ = self.actor_critic.forward(state)
        std_v = torch.sqrt(var_v)
        actions_v = torch.normal(mu_v, std_v)
        # TODO: this might work for angular actuators, but not for
        # prismatic actuators.
        #actions_v = torch.clip(actions_v, min=-0.1, max=0.1)
        self.log_prob = self.compute_log_prob(mu_v, var_v, actions_v)
        return actions_v

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = torch.tensor([state], dtype=torch.float).to(self.actor_critic.device)
        state_ = torch.tensor([state_], dtype=torch.float).to(self.actor_critic.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)

        _, _, critic_value = self.actor_critic.forward(state)
        _, _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()













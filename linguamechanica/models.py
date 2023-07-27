import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(
        self,
        max_action,
        min_variance,
        max_variance,
        lr,
        state_dims,
        action_dims,
        fc1_dims=256,
        fc2_dims=256,
    ):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(*state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, sum(action_dims))
        self.var = nn.Linear(fc2_dims, sum(action_dims))
        self.max_action = max_action
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.mu(x)) * self.max_action
        var = (
            self.min_variance + ((F.tanh(self.var(x)) + 1.0) / 2.0) * self.max_variance
        )
        return mu, var


class Critic(nn.Module):
    def __init__(self, lr, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 
        self.l1 = nn.Linear(sum(state_dim) + sum(action_dim), 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(sum(state_dim) + sum(action_dim), 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

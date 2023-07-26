import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from linguamechanica.dql.replay_memory import ReplayBuffer


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


"""
class Critic(nn.Module):
    def __init__(self, lr, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(sum(state_dim) + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        x = F.tanh(self.l3(x)) * 0.1
        return x
"""


class Critic(nn.Module):
    def __init__(self, lr, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(sum(state_dim) + sum(action_dim), 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
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


class Agent:
    def __init__(
        self,
        lr,
        state_dims,
        action_dims,
        fc1_dims=256,
        fc2_dims=256,
        gamma=0.99,
        mem_size=50000,
    ):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = 0.1
        self.min_variance = 0.001
        self.max_variance = 0.01
        self.noise_clip = 0.1
        self.policy_noise = 0.01
        self.actor = Actor(
            self.max_action,
            self.min_variance,
            self.max_variance,
            lr,
            state_dims,
            action_dims,
            fc1_dims,
            fc2_dims,
        )
        self.critic = Critic(lr, state_dims, action_dims).to(self.actor.device)
        self.actor_target = Actor(
            self.max_action,
            self.min_variance,
            self.max_variance,
            lr,
            state_dims,
            action_dims,
            fc1_dims,
            fc2_dims,
        )
        self.critic_target = Critic(lr, state_dims, action_dims).to(self.actor.device)
        # self.log_prob = None
        self.memory = ReplayBuffer(mem_size, state_dims, action_dims)
        self.total_it = 0
        self.policy_freq = 2
        self.tau = 0.005

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def compute_log_prob(self, mu_v, var_v, actions_v):
        log_prob_part_1 = ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
        log_prob_part_2 = torch.log(torch.sqrt(2 * math.pi * var_v))
        # NOTE: It is addition as it is a multiplication in the non-log domain,
        # but in the log space it is a sum. There is a single probability.
        log_prob = -(log_prob_part_1 + log_prob_part_2)
        return log_prob

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        mu_v, var_v = self.actor.forward(state)
        std_v = torch.sqrt(var_v)
        actions_v = torch.normal(mu_v, std_v)
        """
            TODO: this might work for angular actuators, but not for
            prismatic actuators. It is necessary a noise_clip
            that is congruent with the type of actuator.
        """
        actions_v = torch.clip(actions_v, min=-self.noise_clip, max=self.noise_clip)
        log_prob = self.compute_log_prob(mu_v, var_v, actions_v)
        return actions_v, log_prob

    def learn(self, state, action, log_prob, reward, state_next, done):
        action = torch.tensor([action], dtype=torch.float).to(self.actor.device)
        action = action.unsqueeze(0)
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        state_next = torch.tensor([state_next], dtype=torch.float).to(self.actor.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)

        self.critic.optimizer.zero_grad()
        noise = (torch.randn_like(action) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        mu, var = self.actor(state_next)
        action_next = (mu + noise).clamp(-self.max_action, self.max_action)
        critic_value = self.critic.forward(state, action)
        critic_value_next = self.critic.forward(state, action_next)
        delta = reward + self.gamma * critic_value_next * (1 - int(done)) - critic_value
        critic_loss = delta**2
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        noise = (torch.randn_like(action) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        mu, var = self.actor(state_next)
        action_next = (mu + noise).clamp(-self.max_action, self.max_action)
        critic_value = self.critic.forward(state, action)
        critic_value_next = self.critic.forward(state, action_next)
        delta = reward + self.gamma * critic_value_next * (1 - int(done)) - critic_value
        actor_loss = -log_prob * delta
        actor_loss.backward()
        self.actor.optimizer.step()

    def train(self, state, action, log_prob, reward, next_state, not_done):
        self.total_it += 1
        action = torch.tensor([action], dtype=torch.float).to(self.actor.device)
        action = action.unsqueeze(0)
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        next_state = torch.tensor([next_state], dtype=torch.float).to(self.actor.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        # Sample replay buffer
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_mu, next_var = self.actor_target(next_state)
            next_action = (next_mu + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            mu, var = self.actor(state)
            # TODO: sample using mu and var, and then pass it forward instead of directly mu
            actor_loss = -self.critic.Q1(state, mu).mean()

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def train_buffer(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        action = action.to(self.actor.device)
        state = state.to(self.actor.device)
        next_state = next_state.to(self.actor.device)
        reward = reward.to(self.actor.device)
        not_done = not_done.to(self.actor.device)
        # Sample replay buffer
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_mu, next_var = self.actor_target(next_state)
            next_action = (next_mu + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Qnew = (
                reward.unsqueeze(1) + not_done.unsqueeze(1) * self.gamma * target_Q
            )
        # Optimize the critic
        self.critic.optimizer.zero_grad()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            # Compute actor losse
            mu, var = self.actor(state)
            # TODO: sample using mu and var, and then pass it forward instead of directly mu
            actor_loss = -self.critic.Q1(state, mu).mean()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

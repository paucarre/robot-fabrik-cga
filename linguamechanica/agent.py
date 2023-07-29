import torch
import torch.nn.functional as F
import math
from linguamechanica.models import Actor, Critic
from torchrl.data import ReplayBuffer


class Agent:
    def __init__(
        self,
        lr,
        state_dims,
        action_dims,
        gamma=0.99,
        policy_freq=4,
        tau=0.005,
        max_action=0.1,
        min_variance=0.001,
        max_variance=0.01,
        noise_clip=0.1,
        policy_noise=0.01,
        fc1_dims=256,
        fc2_dims=256,
    ):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
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
        self.replay_buffer = ReplayBuffer()
        self.total_it = 0
        self.policy_freq = policy_freq
        self.tau = tau

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add([state, action, reward, next_state, done])

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

    def train_buffer(self, batch_size=256):
        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        action = action.to(self.actor.device)
        state = state.to(self.actor.device)
        next_state = next_state.to(self.actor.device)
        reward = reward.to(self.actor.device).unsqueeze(1)
        done = done.to(self.actor.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_mu, next_var = self.actor_target(next_state)
            # TODO: next_var is not used, it should e used for sampling and nosing
            next_action = (next_mu + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + (
                (1.0 - done) * self.gamma * torch.min(target_Q1, target_Q2)
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
            """
                #delta = reward + self.gamma * critic_value_next * (1 - int(done)) - critic_value
                #actor_loss = -log_prob * delta
            """
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

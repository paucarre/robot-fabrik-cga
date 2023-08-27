import torch
import torch.nn.functional as F
import math
from linguamechanica.models import IKActor, Critic, PseudoinvJacobianIKActor
from torchrl.data import ReplayBuffer
from dataclasses import asdict
from torchrl.data.replay_buffers import ListStorage


class IKAgent:
    def __init__(
        self,
        open_chain,
        summary,
        lr_actor,
        lr_critic,
        state_dims,
        action_dims,
        gamma=0.99,
        policy_freq=8,
        tau=0.005,
        max_action=1.0,
        min_variance=0.0001,
        max_variance=0.00001,
        policy_noise=0.01,
        fc1_dims=256,
        fc2_dims=256,
        replay_buffer_max_size=10000,
    ):
        self.summary = summary
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.policy_noise = policy_noise
        self.open_chain = open_chain
        self.actor = IKActor(
            open_chain=self.open_chain,
            max_action=self.max_action,
            min_variance=self.min_variance,
            max_variance=self.max_variance,
            lr=lr_actor,
            state_dims=state_dims,
            action_dims=action_dims,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
        )
        self.critic = Critic(
            lr=lr_critic, state_dim=state_dims, action_dim=action_dims
        ).to(self.actor.device)
        self.actor_target = IKActor(
            open_chain=self.open_chain,
            max_action=self.max_action,
            min_variance=self.min_variance,
            max_variance=self.max_variance,
            lr=self.lr_actor,
            state_dims=state_dims,
            action_dims=action_dims,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
        )
        self.jacobian_actor = PseudoinvJacobianIKActor(self.open_chain).to(
            self.actor.device
        )
        self.critic_target = Critic(
            lr=self.lr_critic, state_dim=state_dims, action_dim=action_dims
        ).to(self.actor.device)
        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(max_size=replay_buffer_max_size)
        )
        self.total_it = 0
        self.policy_freq = policy_freq
        self.tau = tau

    def save(self, training_state):
        training_state_dict = asdict(training_state)
        model_dictionary = {
            "critic_target": self.critic_target.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor": self.actor.state_dict(),
        }
        torch.save(
            training_state_dict | model_dictionary,
            f"checkpoints/{training_state.t + 1}.pt",
        )

    def load(self, name):
        data = torch.load(f"checkpoints_{name}.pt")
        self.critic_target.load_state_dict(data["critic_target"])
        self.critic.load_state_dict(data["critic"])
        self.actor.load_state_dict(data["actor"])
        self.actor_target.load_state_dict(data["actor_target"])
        return data

    def store_transition(self, state, action, reward, next_state, done):
        """
        Note that the `replay_buffer` is using a `RoundRobinWriter` and
        thus it will get updated with new data despite the storege
        being full.
        """
        self.replay_buffer.add([state, action, reward, next_state, done])

    def compute_log_prob(self, mu_v, var_v, actions_v):
        log_prob_part_1 = ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
        log_prob_part_2 = torch.log(torch.sqrt(2 * math.pi * var_v))
        # NOTE: It is addition as it is a multiplication in the non-log domain,
        # but in the log space it is a sum. There is a single probability.
        log_prob = -(log_prob_part_1 + log_prob_part_2)
        return log_prob

    def choose_action(self, state, training_state):
        mu_v, var_v = None, None
        state = state.unsqueeze(0).to(self.jacobian_actor.device)
        if training_state.agent_qlearning_training_enabled():
            mu_v, var_v = self.actor(state)
        else:
            mu_v, var_v = self.jacobian_actor(state)
        actions_v = self.sample(mu_v, var_v)
        log_prob = self.compute_log_prob(mu_v, var_v, actions_v)
        return actions_v, log_prob, mu_v, var_v

    def sample(self, mu, var):
        std = torch.sqrt(var)
        actions = torch.normal(mu, std)
        """
            TODO: this might work for angular actuators, but not for
            prismatic actuators. It is necessary a max_action
            that is congruent with the type of actuator.
        """
        actions = torch.clip(actions, min=-self.max_action, max=self.max_action)
        return actions

    def train_buffer(self, training_state):
        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(
            training_state.batch_size()
        )
        self.summary.add_scalar(
            "Train Buffer Action Mean",
            action.mean(),
            training_state.t,
        )
        self.summary.add_scalar(
            "Train Buffer Action Std",
            action.std(),
            training_state.t,
        )

        action = action.to(self.actor.device)
        state = state.to(self.actor.device)
        next_state = next_state.to(self.actor.device)
        reward = reward.to(self.actor.device)
        done = done.to(self.actor.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_mu, next_var = None, None
            if training_state.agent_qlearning_training_enabled():
                next_mu, next_var = self.actor_target(next_state)
            else:
                next_mu, next_var = self.jacobian_actor(next_state)
            next_actions = self.sample(next_mu, next_var)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_actions)
            target_Q = reward + (
                (1.0 - done) * self.gamma * torch.min(target_Q1, target_Q2)
            )
        self.critic.optimizer.zero_grad()
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        self.summary.add_scalar(
            "Critic Loss (Q1 + Q2)",
            critic_loss,
            training_state.t,
        )

        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic.optimizer.step()
        # Delayed policy updates
        if (
            self.total_it % self.policy_freq == 0
            and training_state.agent_qlearning_training_enabled()
        ):
            self.actor.optimizer.zero_grad()
            mu, var = self.actor(state)
            # IMPORTANT NOTE: it is not possible to sample here as it needs to be
            # differentiable
            actor_loss = -self.critic.Q1(state, mu).mean()
            self.summary.add_scalar(
                "Actor Loss",
                actor_loss,
                training_state.t,
            )
            # TODO: entropy loss for the variance missing!!
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()

        if not training_state.agent_qlearning_training_enabled():
            self.actor_target.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            # Compute actor losse
            mu_target, var_target = self.actor_target(state)
            mu, var = self.actor(state)
            # TODO: sample using mu and var, and then pass it forward instead of directly mu
            """
                #delta = reward + self.gamma * critic_value_next * (1 - int(done)) - critic_value
                #actor_loss = -log_prob * delta
            """
            """
            We want to maximize the critic so the closer the 
            critic is to zero, the better.
            """
            det_mu, det_var = self.jacobian_actor(state)
            actor_loss = (mu - det_mu.data).abs().mean()
            actor_target_loss = (mu_target - det_mu.data).abs().mean()
            self.summary.add_scalar(
                "Actor Loss w.r.t. Pseudoinverse Jacobian",
                actor_loss,
                training_state.t,
            )
            self.summary.add_scalar(
                "Actor Target Loss w.r.t. Pseudoinverse Jacobian",
                actor_target_loss,
                training_state.t,
            )
            actor_loss.backward()
            actor_target_loss.backward()
            self.actor_target.optimizer.step()
            self.actor.optimizer.step()

        if (
            self.total_it % self.policy_freq == 0
            and training_state.agent_qlearning_training_enabled()
        ):
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

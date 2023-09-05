import torch
import torch.nn.functional as F
import math
from linguamechanica.models import IKActor, Critic, PseudoinvJacobianIKActor
from torchrl.data import ReplayBuffer
from dataclasses import asdict
import torch.optim as optim
from torchrl.data.replay_buffers import ListStorage
#from torchrl.data.replay_buffers import TensorStorage
from enum import Enum
from enum import auto


class AgentState(Enum):
    JACBOBIAN_TRANING = auto()
    QLEARNING_TRANING = auto()


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
        max_noise_clip=0.0001,
        initial_action_variance=0.0001,
        max_variance=0.00001,
        policy_noise=0.01,
        fc1_dims=256,
        fc2_dims=256,
        replay_buffer_max_size=10000,
    ):
        self.summary = summary
        self.max_noise_clip = max_noise_clip
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.initial_action_variance = initial_action_variance
        self.max_variance = max_variance
        self.policy_noise = policy_noise
        self.open_chain = open_chain
        self.actor = IKActor(
            open_chain=self.open_chain,
            max_action=self.max_action,
            initial_action_variance=self.initial_action_variance,
            max_variance=self.max_variance,
            lr=lr_actor,
            action_dims=action_dims,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
        )
        self.critic = Critic(
            lr=lr_critic,
            action_dims=action_dims,
            open_chain=open_chain,
        ).to(self.actor.device)
        self.actor_target = IKActor(
            open_chain=self.open_chain,
            max_action=self.max_action,
            initial_action_variance=self.initial_action_variance,
            max_variance=self.max_variance,
            lr=self.lr_actor,
            action_dims=action_dims,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
        )
        self.jacobian_actor = PseudoinvJacobianIKActor(self.open_chain).to(
            self.actor.device
        )
        self.critic_target = Critic(
            lr=lr_critic,
            action_dims=action_dims,
            open_chain=open_chain,
        ).to(self.actor.device)
        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(max_size=replay_buffer_max_size)
        )
        self.total_it = 0
        self.policy_freq = policy_freq
        self.tau = tau
        self.create_optimizers()
        self.state = AgentState.QLEARNING_TRANING

    def create_optimizers(self):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        # self.actor_target_optimizer = optim.Adam(self.actor_target.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        # self.critic_target_optimizer = optim.Adam(self.critic_target.parameters(), lr=self.lr_critic)

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
        print("store_transition", state.shape, action.shape, reward.shape, next_state.shape, done.shape) 
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
        state = state.to(self.jacobian_actor.device)
        #if self.state == AgentState.QLEARNING_TRANING:
        mu_v, var_v = self.actor(state)
        #else:
        #    mu_v, var_v = self.jacobian_actor(state)
        #TODO: make (1, 6) constant parametrizable
        '''
        mu_v = torch.zeros(1, 6).to(self.jacobian_actor.device)
        var_v = (
            torch.normal(torch.zeros(1, 6), torch.ones(6) * 0.0001)
            .to(self.jacobian_actor.device)
            .abs()
        )
        '''
        actions_v, noise = self.sample(mu_v, var_v)
        log_prob = self.compute_log_prob(mu_v, var_v, actions_v)
        return actions_v, log_prob, mu_v, noise

    def sample(self, mu, var):
        std = torch.sqrt(var)
        noise = torch.randn_like(mu) * std
        """
            TODO: this might work for angular actuators, but not for
            prismatic actuators. It is necessary a max_noise_clip
            that is congruent with the type of actuator.
        """
        noise = torch.clip(noise, min=-self.max_noise_clip, max=self.max_noise_clip)
        actions = mu + noise 
        return actions, noise

    def train_buffer(self, training_state):
        self.state = AgentState.QLEARNING_TRANING
        '''
        if (
            training_state.can_train_buffer()
            and self.state == AgentState.JACBOBIAN_TRANING
        ):
            # reset optimizers
            self.create_optimizers()
            self.state = AgentState.QLEARNING_TRANING
        '''

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
            if self.state == AgentState.QLEARNING_TRANING:
                print("if self.state == AgentState.QLEARNING_TRANING", next_state.shape)
                next_mu, next_var = self.actor_target(next_state)
            else:
                next_mu, next_var = self.jacobian_actor(next_state)
            next_actions, noise = self.sample(next_mu, next_var)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_actions)
            target_Q = reward + (
                (1.0 - done) * self.gamma * torch.min(target_Q1, target_Q2)
            )
        self.critic_optimizer.zero_grad()
        current_Q1, current_Q2 = self.critic(state, action)
        quality_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        quality_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        self.summary.add_scalar(
            "Quality Loss (Q1 + Q2)",
            quality_loss,
            training_state.t,
        )

        # Delayed policy updates
        if (
            self.total_it % self.policy_freq == 0
            #quality_loss.item() < 0.1
            and self.state == AgentState.QLEARNING_TRANING
        ):
            self.actor_optimizer.zero_grad()
            mu, var = self.actor(state)
            # IMPORTANT NOTE: it is not possible to sample here as it needs to be
            # differentiable, thus the variance is ignored and not used
            actor_loss = -self.critic.Q1(state, mu).mean()
            self.summary.add_scalar(
                "Actor Loss",
                actor_loss,
                training_state.t,
            )
            # TODO: entropy loss for the variance missing!!
            actor_loss.backward()
            #x torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

        if self.state == AgentState.JACBOBIAN_TRANING:
            self.actor_optimizer.zero_grad()
            # Compute actor losse
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
            self.summary.add_scalar(
                "Actor Loss w.r.t. Pseudoinverse Jacobian",
                actor_loss,
                training_state.t,
            )
            actor_loss.backward()
            self.actor_optimizer.step()

        if self.state == AgentState.JACBOBIAN_TRANING:
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
                target_param.data.copy_(param.data)
        elif (
            self.total_it % self.policy_freq == 0
            and self.state == AgentState.QLEARNING_TRANING
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

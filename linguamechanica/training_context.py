from dataclasses import dataclass
import torch


@dataclass
class TrainingState:
    save_freq: int = 1000
    lr_actor: float = 0.00001
    lr_critic: float = 0.000001
    gamma: float = 0.99
    policy_freq: int = 32
    tau: float = 0.01
    eval_freq: int = 200
    max_timesteps: float = 1e6
    data_generation_without_actor_iterations: int = 1000
    qlearning_batch_size: int = 1024  # 32
    batch_size_jacobian: int = 32
    jacobian_batch_size = 16
    min_variance: int = 1e-5
    max_variance: int = 2e-4
    max_noise_clip: int = 2e-4
    max_action: int = 0.2
    t: int = 0
    weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # NOTE: it should have enough iterations for the replay buffer
    # to have enough data to train the actor
    # actor_qlearning_training_starts_at_iteration = 2000
    jacobian_learning_iterations = 3000  # 20000
    # Initially actor training is disable only the critic
    # is trained because the actor only uses the jacobian
    # actor_training_enabled: bool = False
    max_steps_done: int = 20
    max_episodes_in_buffer:int = 50

    def replay_buffer_max_size(self):
        return self.max_episodes_in_buffer * self.max_steps_done

    def can_train_buffer(self):
        return self.t >= self.data_generation_without_actor_iterations

    def agent_qlearning_training_enabled(self):
        return (
            self.t
            >= self.data_generation_without_actor_iterations
            + self.jacobian_learning_iterations
        )

    def use_actor_for_data_generation(self):
        return self.t >= self.data_generation_without_actor_iterations

    def can_save(self):
        return (
            self.t + 1
        ) % self.save_freq == 0 and self.agent_qlearning_training_enabled()

    def can_eval_policy(self):
        return (
            self.t + 1
        ) % self.eval_freq == 0 and self.agent_qlearning_training_enabled()

    def batch_size(self):
        if self.agent_qlearning_training_enabled():
            return self.qlearning_batch_size
        else:
            return self.jacobian_batch_size


@dataclass
class EpisodeState:
    reward = 0
    timesteps = 0
    num = 0
    gamma = 0
    discounted_gamma = 0

    def __init__(self, gamma):
        self.gamma = gamma
        self.discounted_gamma = gamma

    def step(self, reward):
        self.reward = self.reward + (self.discounted_gamma * reward)
        self.discounted_gamma *= self.gamma

    def create_new(self):
        self.reward = 0
        self.timesteps = 0
        self.num += 1
        self.discounted_gamma = self.gamma


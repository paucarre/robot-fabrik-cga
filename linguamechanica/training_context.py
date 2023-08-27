from dataclasses import dataclass
import torch


@dataclass
class TrainingState:
    save_freq: int = 1000
    lr_actor: float = 0.00001
    lr_critic: float = 0.00001
    gamma: float = 0.99
    policy_freq: int = 8
    tau: float = 0.05
    eval_freq: int = 200
    max_action: int = 0.2
    max_timesteps: float = 1e6
    data_generation_without_actor_iterations: int = 1000
    qlearning_batch_size: int = 1024  # 32
    batch_size_jacobian: int = 32
    jacobian_batch_size = 16
    min_variance: int = 0.00001
    max_variance: int = 0.00001
    t: int = 0
    weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # NOTE: it should have enough iterations for the replay buffer
    # to have enough data to train the actor
    # actor_qlearning_training_starts_at_iteration = 2000
    jacobian_learning_iterations = 1000  # 20000
    # Initially actor training is disable only the critic
    # is trained because the actor only uses the jacobian
    # actor_training_enabled: bool = False
    max_steps_done: int = 20
    replay_buffer_max_size: int = 1000

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

    def create_new(self):
        self.reward = 0
        self.timesteps = 0
        self.num += 1

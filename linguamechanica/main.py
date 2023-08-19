import numpy as np
import torch
from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass


@dataclass
class TrainingState:
    save_freq: int = 1000
    lr_actor: float = 0.0001
    lr_critic: float = 0.05
    gamma: float = 0.99
    policy_freq: int = 16
    tau: float = 0.05
    eval_freq: int = 200
    max_timesteps: float = 1e6
    start_timesteps: int = 20
    batch_size: int = 32
    jacobian_reduction: float = 1e-4
    t: int = 0
    jacobian_proportion: float = 1.0
    weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def can_save(self):
        return (self.t + 1) % self.save_freq == 0 and self.t >= self.start_timesteps

    def can_eval_policy(self):
        return (self.t + 1) % self.eval_freq == 0 and self.t >= self.start_timesteps

    def jacobian_proportion_step(self):
        self.jacobian_proportion -= self.jacobian_reduction
        self.jacobian_proportion = max(0.0, self.jacobian_proportion)
        if self.jacobian_proportion == 0.0:
            self.batch_size = 1024


@dataclass
class EpisodeState:
    reward = 0
    timesteps = 0
    num = 0

    def create_new(self):
        self.reward = 0
        self.timesteps = 0
        self.num += 1


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(agent, weights, eval_episodes=10):
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chain = urdf_robot.extract_open_chains(0.3)[-1]
    eval_env = Environment(open_chain, weights)

    avg_acc_reward = 0.0
    initial_rewards = torch.zeros(eval_episodes)
    final_rewards = torch.zeros(eval_episodes)
    for idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        eval_reward = None
        reward = None
        while not done:
            action, log_prob = agent.choose_action(np.array(state))
            state, reward, done = eval_env.step(action)
            if eval_reward is None:
                initial_rewards[idx] = reward
                eval_reward = reward
            else:
                eval_reward += reward
        final_rewards[idx] = reward
        avg_acc_reward += eval_reward
    avg_acc_reward /= eval_episodes

    print(f"Evaluation over {eval_episodes} episodes: {avg_acc_reward.item():.3f}")
    return avg_acc_reward, initial_rewards, final_rewards


def summary_evaluatation(
    summary, initial_rewards, final_rewards, avg_acc_reward, training_state
):
    # summary.add_scalar("Avg Initial Reward Eval", initial_rewards.mean(), t)
    # summary.add_scalar("Avg Final Reward Eval", final_rewards.mean(), t)
    summary.add_scalar(
        "Avg Improved Reward Eval",
        (final_rewards / (initial_rewards + 1e-10)).mean(),
        training_state.t,
    )
    summary.add_scalar("Accumulated Reward Eval", avg_acc_reward, training_state.t)
    summary.add_scalar(
        "Jacobian Proportion", training_state.jacobian_proportion, training_state.t
    )


def summary_done(summary, training_state, episode):
    # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
    print(
        f"Total T: {training_state.t+1} Episode Num: {episode.num+1} Episode T: {episode.timesteps} Reward: {episode.reward.item():.3f}"
    )
    summary.add_scalar("Reward/train", episode.reward, training_state.t)


def main():
    summary = SummaryWriter()
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chain = urdf_robot.extract_open_chains(0.3)[-1]
    # TODO: place all these constants as arguments

    training_state = TrainingState()
    env = Environment(open_chain, training_state.weights)
    agent = IKAgent(
        open_chain=open_chain,
        lr_actor=training_state.lr_actor,
        lr_critic=training_state.lr_critic,
        state_dims=(env.observation_space.shape),
        action_dims=env.action_dims,
        gamma=training_state.gamma,
        policy_freq=training_state.policy_freq,
        tau=training_state.tau,
    )
    episode = EpisodeState()
    state, done = env.reset(), False
    initial_reward = None
    # agent.load(f"checkpoint_46999")

    for training_state.t in range(training_state.t, int(training_state.max_timesteps)):
        # print(f"Current timestamp: {t}")
        episode.timesteps += 1
        # Select action randomly or according to policy
        if training_state.t < training_state.start_timesteps:
            action = env.sample_random_action()
        else:
            action, log_prob = agent.choose_action(np.array(state))

        # Perform action
        next_state, reward, done = env.step(action)
        # TODO: I think this should be in the environment, I see no reason
        # why this should be kept here
        done = float(done) if episode.timesteps < env.max_steps_done else 1
        agent.store_transition(
            state=state.detach().cpu(),
            action=action[0, :].detach().cpu(),
            reward=reward.detach().cpu(),
            next_state=next_state.detach().cpu(),
            done=torch.Tensor([done]).detach().cpu(),
        )

        state = next_state
        episode.reward += reward
        if initial_reward is None:
            initial_reward = reward

        # Train agent after collecting sufficient data
        if training_state.t >= training_state.start_timesteps:
            agent.train_buffer(training_state, summary)
        if done and training_state.t >= training_state.start_timesteps:
            summary_done(summary, training_state, episode)
        if training_state.can_eval_policy():
            avg_acc_reward, initial_rewards, final_rewards = eval_policy(
                agent, training_state.weights, 2
            )
            summary_evaluatation(
                summary, initial_rewards, final_rewards, avg_acc_reward, training_state
            )

        if training_state.can_save():
            agent.save(
                training_state.t, training_state, f"checkpoint_{training_state.t + 1}"
            )

        if done:
            final_reward = reward
            summary.add_scalar(
                "Reward Improvement Training",
                final_reward / (initial_reward + 1e-10),
                training_state.t,
            )
            state = env.reset()
            initial_reward = None
            done = False
            episode.create_new()

        training_state.jacobian_proportion_step()

    summary.close()


if __name__ == "__main__":
    main()

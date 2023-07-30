import numpy as np
import torch
from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import Agent
from torch.utils.tensorboard import SummaryWriter


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(agent, eval_episodes=10):
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.3)
    eval_env = Environment(open_chains[-1])

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action, log_prob = agent.choose_action(np.array(state))
            state, reward, done = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def main():
    tensorbard_summary = SummaryWriter()

    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.3)
    env = Environment(open_chains[-1])
    agent = Agent(
        lr=0.000001,
        state_dims=(env.observation_space.shape),
        action_dims=env.action_dims,
        gamma=0.99,
        policy_freq=8,
        tau=0.005,
    )
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_freq = 200
    max_timesteps = 1e6
    start_timesteps = 2000
    batch_size = 1024
    for t in range(int(max_timesteps)):
        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.sample_random_action()
        else:
            action, log_prob = agent.choose_action(np.array(state))

        # Perform action
        next_state, reward, done = env.step(action)
        # TODO: I think this should be in the environment, I see no reason
        # why this should be kept here
        done = float(done) if episode_timesteps < env.max_steps_done else 1
        agent.store_transition(
            state=state.detach().cpu(),
            action=action[0, :].detach().cpu(),
            reward=reward.detach().cpu(),
            next_state=next_state.detach().cpu(),
            done=torch.Tensor([done]).detach().cpu(),
        )

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            agent.train_buffer(batch_size)
        """
            LOGGING
            TODO: move this in its own method
        """
        if done and t >= start_timesteps:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            tensorbard_summary.add_scalar("Reward/train", episode_reward, t)
        if (t + 1) % eval_freq == 0 and t >= start_timesteps:
            average_reward = eval_policy(agent, 2)
            tensorbard_summary.add_scalar("Reward/evaluation", average_reward, t)

        if done:
            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    tensorbard_summary.close()


if __name__ == "__main__":
    main()

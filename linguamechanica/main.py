import numpy as np
import torch
from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import Agent
from torchrl.data import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import traceback
import logging
import pdb

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

    replay_buffer = ReplayBuffer()  # state_dim, action_dim)
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.3)
    env = Environment(open_chains[-1])
    agent = Agent(
        lr=0.000001,
        state_dims=(env.observation_space.shape),
        action_dims=env.action_dims,
        gamma=0.99,
        policy_freq=4,
        tau=0.005,
    )
    evaluations = [eval_policy(agent)]
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_freq = 200
    max_timesteps = 1e6
    start_timesteps = 2000
    batch_size = 32
    for t in range(int(max_timesteps)):
        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.sample_random_action()
        else:
            action, log_prob = agent.choose_action(np.array(state))

        # Perform action
        next_state, reward, done = env.step(action)
        done_bool = float(done) if episode_timesteps < env.max_steps_done else 0

        replay_buffer.add(
            [
                torch.Tensor(state).cpu(),
                action[0, :].detach().cpu(),
                torch.Tensor(next_state).cpu(),
                reward.detach().cpu(),
                torch.Tensor([done_bool]).cpu(),
            ]
        )

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            agent.train_buffer(replay_buffer, batch_size)
        if done and t >= start_timesteps:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            tensorbard_summary.add_scalar("Reward/train", episode_reward, t)
            tensorbard_summary.flush()
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % eval_freq == 0 and t >= start_timesteps:
            average_reward = eval_policy(agent, 2)
            evaluations.append(average_reward)
            tensorbard_summary.add_scalar("Reward/evaluation", average_reward, t)
            tensorbard_summary.flush()
            # np.save(f"./results/{file_name}", evaluations)
            # if args.save_model: policy.save(f"./models/{file_name}")
    tensorbard_summary.close()


if __name__ == "__main__":
    try:
        pdb.set_trace() 
        main()
    except:
        logging.error(traceback.format_exc())

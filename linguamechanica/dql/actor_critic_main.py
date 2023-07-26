import numpy as np
import random
import math
import torch
from pytorch3d import transforms
from linguamechanica.dql.actor_critic import Agent
from linguamechanica.dql.utils import plot_learning_curve
from linguamechanica.kinematics import UrdfRobotLibrary


class Environment:
    def __init__(self, open_chain):
        self.open_chain = open_chain
        """
        State dims should be for now:
            - Target pose, 6 
            - Current parameters, 6
            - Current parameter index one hot, 6
        Action size should be:
            - Angle: sigmoid(x) - 0.5 or something similar
        """
        self.observation_space = np.zeros(6 + 6)
        self.action_dims = np.zeros(6).shape
        self.current_step = 0
        self._max_episode_steps = 200

    def uniformly_sample_parameters_within_constraints(self):
        coordinates = []
        for i in range(len(self.open_chain.joint_limits)):
            coordinates.append(
                random.uniform(
                    self.open_chain.joint_limits[i][0],
                    self.open_chain.joint_limits[i][1],
                )
            )
        return torch.Tensor(coordinates).unsqueeze(0)

    def observation_to_numpy(self, observation):
        observation_np = np.zeros(*self.observation_space.shape, dtype=np.float32)
        observation_np[:6] = observation["target_pose"]
        # TODO: test if this is key for backpropagation on manifold.
        # It should be a pytorch tensor and undetached
        observation_np[6:12] = observation["current_pose"].detach().numpy()
        # index = torch.Tensor(6)
        # index[observation["current_parameter_index"]] = 1.0
        # observation_np[12:] = index[:]
        return observation_np

    def sample_random_action(self):
        # TODO: this is a bit silly for now
        return self.uniformly_sample_parameters_within_constraints() / math.pi

    def generate_observation(self):
        self.current_transformation = self.open_chain.forward_transformation(
            self.current_parameters
        )
        self.current_pose = transforms.se3_log_map(
            self.current_transformation.transpose(1, 2)
        )
        self.current_parameter_index = 0
        observation = {
            "target_transformation": self.target_transformation,
            "target_parameters": self.target_parameters,
            "target_pose": self.target_pose,
            "current_parameters": self.current_parameters,
            "current_transformation": self.current_transformation,
            "current_pose": self.current_pose,
            "current_parameter_index": self.current_parameter_index,
        }
        observation = self.observation_to_numpy(observation)
        return observation

    def reset(self):
        self.target_parameters = self.uniformly_sample_parameters_within_constraints()
        self.target_transformation = self.open_chain.forward_transformation(
            self.target_parameters
        )
        self.target_pose = transforms.se3_log_map(
            self.target_transformation.transpose(1, 2)
        )
        self.current_parameters = self.uniformly_sample_parameters_within_constraints()
        observation = self.generate_observation()
        self.current_step = 0
        return observation

    def compute_reward(self):
        target_pose = transforms.Transform3d(
            matrix=transforms.se3_exp_map(self.target_parameters)
        )
        current_pose = transforms.Transform3d(
            matrix=transforms.se3_exp_map(self.current_parameters)
        )
        pose_difference = target_pose.inverse().compose(current_pose)
        # TODO: this needs reweighting of angles and distances
        pose_distance = pose_difference.get_se3_log().abs().sum()
        # TODO: this needs to be implemented properly
        done = pose_distance < 1e-1
        return -pose_distance, done

    def step(self, action):
        self.current_step += 1
        # print(f"Action {action}, {self.current_parameter_index}")
        # TODO: make action indices generic
        # print(self.current_parameters.shape, action.shape)
        self.current_parameters[:, :] += action[:, :].cpu()
        # self.current_parameter_index = (self.current_parameter_index + 1) % len(
        #    self.open_chain
        # )
        # print(f"{self.target_parameters} | {self.current_parameters}")
        reward, done = self.compute_reward()
        observation = self.generate_observation()
        if self.current_step >= self._max_episode_steps:
            done = 1
        return observation, reward, done


def main():
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.3)
    env = Environment(open_chains[-1])  # make_env("PongNoFrameskip-v4")
    load_checkpoint = False
    n_games = 50
    agent = Agent(
        lr=0.00000001,
        state_dims=(env.observation_space.shape),
        n_actions=env.action_space,
    )

    if load_checkpoint:
        agent.load_models()

    fname = "inverse_kinematics" + "_" + str(n_games) + "games"
    figure_file = "plots/" + fname + ".png"

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    scores = []
    # For each IK problem
    for i in range(n_games):
        done = False
        state = env.reset()
        score = 0
        # Loop to solve IK
        while not done:
            """
            Given the state, get the action from the actor network.
            Also compute the log probabilities in the agent state
            """
            action, log_prob = agent.choose_action(state)
            """
                Apply the action to the robot and get from it
                the next state, the reward and whether the
                IK problem is solved
            """
            state_next, reward, done = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(state, action, reward, state_next, int(done))
            """
                Train loop for actor and policy network
            """
            agent.train(state, action, log_prob, reward, state_next, (1 - int(done)))
            state = state_next
            n_steps += 1
            if n_steps % 100 == 0:
                print(f"\t{n_steps} step | Reward {reward} | Score {score}")
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print("episode ", i, "score %.1f" % score, "average score %.1f" % avg_score)

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)


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
            mu, var = agent.choose_action(np.array(state))
            # TODO: sample using mu and var and not directly mu
            state, reward, done = eval_env.step(mu)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def main2():
    from torchrl.data import ReplayBuffer

    replay_buffer = ReplayBuffer()  # state_dim, action_dim)
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.3)
    env = Environment(open_chains[-1])
    agent = Agent(
        lr=0.00000001,
        state_dims=(env.observation_space.shape),
        action_dims=env.action_dims,
    )
    # Evaluate untrained policy
    evaluations = [eval_policy(agent, 1)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_freq = 100
    max_timesteps = 1e4
    start_timesteps = 200
    max_action = 0.1
    batch_size = 32
    for t in range(int(max_timesteps)):
        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.sample_random_action()
        else:
            mu, var = agent.choose_action(np.array(state))
            # TODO: clip action
            std = torch.sqrt(var)
            action = torch.clip(torch.normal(mu, std), -max_action, max_action)

        # Perform action
        next_state, reward, done = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        # breakpoint()
        """
        state and next_state are not proper tensors, numpy
        done_bool is a number
        reward has no batch size
        """
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

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_policy(agent, 1))
            # np.save(f"./results/{file_name}", evaluations)
            # if args.save_model: policy.save(f"./models/{file_name}")


if __name__ == "__main__":
    main2()

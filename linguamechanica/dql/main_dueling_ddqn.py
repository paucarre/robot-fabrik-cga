import numpy as np
import random
import torch
from pytorch3d import transforms
from linguamechanica.dql.dueling_ddqn_agent import DuelingDDQNAgent
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
        self.observation_space = np.zeros(6 + 6 + 6)
        self.action_space = 1

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
        observation_np[:6] = observation["target_parameters"]
        observation_np[6:12] = observation["current_parameters"]
        index = torch.Tensor(6)
        index[observation["current_parameter_index"]] = 1.0
        observation_np[12:] = index[:]
        return observation_np

    def reset(self):
        self.target_parameters = self.uniformly_sample_parameters_within_constraints()
        self.target_transformation = self.open_chain.forward_transformation(
            self.target_parameters
        )
        self.target_pose = transforms.se3_log_map(
            self.target_transformation.transpose(1, 2)
        )
        self.current_parameters = self.uniformly_sample_parameters_within_constraints()
        self.current_parameter_index = 0
        observation = {
            "target_transformation": self.target_transformation,
            "target_parameters": self.target_parameters,
            "target_pose": self.target_pose,
            "current_parameters": self.current_parameters,
            "current_parameter_index": self.current_parameter_index,
        }
        observation = self.observation_to_numpy(observation)
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
        return 1.0 / (1.0 + pose_distance), done

    def step(self, action):
        # print(f"Action {action}, {self.current_parameter_index}")
        self.current_parameters[0, self.current_parameter_index] += action
        self.current_parameter_index = (self.current_parameter_index + 1) % len(
            self.open_chain
        )
        # print(f"{self.target_parameters} | {self.current_parameters}")
        reward, done = self.compute_reward()
        return observation, reward, done


if __name__ == "__main__":
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.3)
    env = Environment(open_chains[-1])  # make_env("PongNoFrameskip-v4")
    best_score = -np.inf
    load_checkpoint = False
    n_games = 20
    agent = DuelingDDQNAgent(
        gamma=0.99,
        epsilon=1.0,
        lr=0.000001,
        state_dims=(env.observation_space.shape),
        n_actions=env.action_space,
        mem_size=50000,
        eps_min=0.1,
        batch_size=32,
        replace=10000,
        eps_dec=1e-5,
        chkpt_dir="models/",
        algo="DuelingDDQNAgent",
        env_name="PongNoFrameskip-v4",
    )

    if load_checkpoint:
        agent.load_models()

    fname = (
        agent.algo
        + "_"
        + agent.env_name
        + "_lr"
        + str(agent.lr)
        + "_"
        + str(n_games)
        + "games"
    )
    figure_file = "plots/" + fname + ".png"

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        print(f"Playing game {i + 1}/{n_games}")
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(
                    observation, action, reward, observation_, int(done)
                )
            agent.learn()
            observation = observation_
            n_steps += 1
            if n_steps % 100 == 0:
                print(f"\t{n_steps} step | Reward {reward} | Score {score}")
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print(
            "episode: ",
            i,
            "score: ",
            score,
            " average score %.1f" % avg_score,
            "best score %.2f" % best_score,
            "epsilon %.2f" % agent.epsilon,
            "steps",
            n_steps,
        )

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

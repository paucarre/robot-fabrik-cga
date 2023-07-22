import numpy as np
import random
import torch
from pytorch3d import transforms
from linguamechanica.dql.actor_critic import Agent, ActorCriticNetwork
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
        observation_np[:6] = observation["target_pose"]
        # TODO: test if this is key for backpropagation on manifold. 
        # It should be a pytorch tensor and undetached
        observation_np[6:12] = observation["current_pose"].detach().numpy()
        #index = torch.Tensor(6)
        #index[observation["current_parameter_index"]] = 1.0
        #observation_np[12:] = index[:]
        return observation_np

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
        return observation

    def compute_reward(self):
        target_pose  = transforms.Transform3d(matrix=transforms.se3_exp_map(self.target_parameters))
        current_pose = transforms.Transform3d(matrix=transforms.se3_exp_map(self.current_parameters))
        pose_difference = target_pose.inverse().compose(current_pose)
        #TODO: this needs reweighting of angles and distances
        pose_distance = pose_difference.get_se3_log().abs().sum() 
        #TODO: this needs to be implemented properly
        done = pose_distance < 1e-1 
        return -pose_distance, done

    def step(self, action):
        #print(f"Action {action}, {self.current_parameter_index}")
        # TODO: make action indices generic
        #print(self.current_parameters.shape, action.shape)
        self.current_parameters[0, :] += action[0, :].cpu()
        #self.current_parameter_index = (self.current_parameter_index + 1) % len(
        #    self.open_chain
        #)
        #print(f"{self.target_parameters} | {self.current_parameters}")
        reward, done = self.compute_reward()
        observation = self.generate_observation()
        return observation, reward, done

def main():
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.3)
    env = Environment(open_chains[-1])  # make_env("PongNoFrameskip-v4")
    best_score = -np.inf
    load_checkpoint = False
    n_games = 20
    agent = Agent(
        lr=0.000001,
        state_dims=(env.observation_space.shape),
        n_actions=env.action_space
    )

    if load_checkpoint:
        agent.load_models()

    fname = ("inverse_kinematics"
        + "_"
        + str(n_games)
        + "games"
    )
    figure_file = "plots/" + fname + ".png"

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(
                    observation, action, reward, observation_, int(done)
                )
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            n_steps += 1
            if n_steps % 100 == 0:
                print(f"\t{n_steps} step | Reward {reward} | Score {score}")
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)


if __name__ == '__main__':
    main()
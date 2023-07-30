import numpy as np
import random
import torch
from pytorch3d import transforms
import math


class Environment:
    def __init__(self, open_chain, max_steps_done=200):
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
        self.max_steps_done = max_steps_done

    def uniformly_sample_parameters_within_constraints(self):
        coordinates = []
        for i in range(len(self.open_chain.joint_limits)):
            # TODO: check if unconstrained works
            coordinates.append(
                random.uniform(
                    self.open_chain.joint_limits[i][0],
                    self.open_chain.joint_limits[i][1],
                )
            )
        return torch.Tensor(coordinates).unsqueeze(0)

    def observation_to_tensor(self, observation):
        observation_tensor = torch.zeros(self.observation_space.shape)
        observation_tensor[:6] = observation["target_pose"].detach().cpu()
        # TODO: test if this is key for backpropagation on manifold.
        # It should be a pytorch tensor and undetached
        observation_tensor[6:12] = observation["current_pose"].detach().cpu()
        # index = torch.Tensor(6)
        # index[observation["current_parameter_index"]] = 1.0
        # observation_np[12:] = index[:]
        return observation_tensor

    def sample_random_action(self):
        # TODO: this is a bit silly for now
        return self.uniformly_sample_parameters_within_constraints() / math.pi

    def generate_observation(self):
        self.current_transformation = self.open_chain.forward_transformation(
            self.current_parameters
        )
        self.current_pose = transforms.se3_log_map(
            self.current_transformation.get_matrix()
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
        observation = self.observation_to_tensor(observation)
        return observation

    def reset(self):
        self.target_parameters = self.uniformly_sample_parameters_within_constraints()
        self.target_transformation = self.open_chain.forward_transformation(
            self.target_parameters
        )
        self.target_pose = transforms.se3_log_map(
            self.target_transformation.get_matrix()
        )
        # TODO: add a level, max noise clip and
        # also restrict to constraints
        # neecessary also be able to have levels....
        noise = torch.randn_like(self.target_parameters)
        noise = noise.clamp(-0.1, 0.1)
        self.current_parameters = self.target_parameters + (noise)
        # self.uniformly_sample_parameters_within_constraints()
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
        """
        This means the pose of the current pose wrt the 
        target pose.
        Note that the `compose` method is left-application of
        a left-matrix, meaning that it is equivalnet to:
        `target_pose.inverse() @ current_pose`
        """
        pose_difference = current_pose.compose(target_pose.inverse())
        # TODO: this needs reweighting of angles and distances
        # TODO: currently it only accounts for translation ("[:3]"), not rotation
        pose_distance = pose_difference.get_se3_log()[:3].abs().sum()
        # TODO: this needs to be implemented properly
        done = pose_distance < 1e-1
        return -pose_distance, done

    def step(self, action):
        self.current_step += 1
        # print(f"Action {action}, {self.current_parameter_index}")
        self.current_parameters[:, :] += action[:, :].cpu()
        # self.current_parameter_index = (self.current_parameter_index + 1) % len(
        #    self.open_chain
        # )
        # print(f"{self.target_parameters} | {self.current_parameters}")
        reward, done = self.compute_reward()
        observation = self.generate_observation()
        if self.current_step >= self.max_steps_done:
            done = 1
        return observation, reward, done

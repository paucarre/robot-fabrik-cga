import numpy as np
import random
import torch
from pytorch3d import transforms
import math
from linguamechanica.kinematics import DifferentiableOpenChainMechanism


class Environment:
    def __init__(self, open_chain, weights, max_steps_done=200):
        self.open_chain = open_chain
        """
        State dims should be for now:
            - Target pose, 6 
            - Current pose, 6
            - Current parameters, 6 
              Note that the current pose might
              not be informative enough as to know
              the current parameters one would need to solve the
              inverse kinematics for the current pose,
              which might be an even more difficult task.
        Action size should be:
            - Angle: sigmoid(x) - 0.5 or something similar
        """
        self.weights = weights
        self.observation_space = np.zeros(6 + 6)
        self.action_dims = np.zeros(6).shape
        self.current_step = 0
        self.max_steps_done = max_steps_done
        # TODO: make this nicer
        self.device = "cuda:0"
        self.open_chain = self.open_chain.to(self.device)
        self.weights = self.weights.to(self.device)

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
        return torch.Tensor(coordinates).unsqueeze(0).to(self.device)

    def observation_to_tensor(self, observation):
        observation_tensor = torch.zeros(self.observation_space.shape)
        observation_tensor[:6] = observation["target_pose"].detach().cpu()
        # TODO: test if this is key for backpropagation on manifold.
        # It should be a pytorch tensor and undetached
        # observation_tensor[6:12] = observation["current_pose"].detach().cpu()
        observation_tensor[6:] = observation["current_parameters"].detach().cpu()
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
        # necessary also be able to have levels....
        noise = torch.randn_like(self.target_parameters)
        noise = noise.clamp(-0.1, 0.1)
        self.current_parameters = (self.target_parameters + (noise)).to(self.device)
        # self.uniformly_sample_parameters_within_constraints()
        observation = self.generate_observation()
        self.current_step = 0
        return observation

    def compute_reward(self):
        error_pose = self.open_chain.compute_error_pose(
            self.current_parameters, self.target_pose
        )
        pose_error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_pose, self.weights
        )
        done = pose_error < 1e-1
        return -pose_error, done

    def step(self, action):
        self.current_step += 1
        # print(f"Action {action}, {self.current_parameter_index}")
        """
        TODO:
        Clip the current parameters to the max and min values.
        Even if there are no constraints this is necessary. For
        instance, the revolute joints will go from (-pi, pi)
        or (0, 2 * pi).
        """
        self.current_parameters[:, :] += action[:, :]
        # self.current_parameter_index = (self.current_parameter_index + 1) % len(
        #    self.open_chain
        # )
        # print(f"{self.target_parameters} | {self.current_parameters}")
        reward, done = self.compute_reward()
        observation = self.generate_observation()
        if self.current_step >= self.max_steps_done:
            done = 1
        return observation, reward, done

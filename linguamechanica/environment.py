import numpy as np
import random
import torch
from pytorch3d import transforms
import math
from linguamechanica.kinematics import DifferentiableOpenChainMechanism


def force_parameters_within_bounds(params):
    bigger_than_pi = params[:, :] > math.pi
    params[bigger_than_pi] = params[bigger_than_pi] - (2.0 * math.pi)
    less_than_minus_pi = params[:, :] < -math.pi
    params[less_than_minus_pi] = params[less_than_minus_pi] + (2.0 * math.pi)


class Environment:
    def __init__(self, open_chain, training_state):
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
        self.weights = training_state.weights
        self.observation_space = np.zeros(6 + 6)
        self.action_dims = np.zeros(6).shape
        self.current_step = 0
        self.max_steps_done = training_state.max_steps_done
        # TODO: make this nicer
        self.device = "cuda:0"
        self.open_chain = self.open_chain.to(self.device)
        self.weights = self.weights.to(self.device)
        self.max_noise_clip = training_state.max_noise_clip
        self.max_action = training_state.max_action

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

    def sample_random_action(self):
        # TODO: this is a bit silly for now
        return self.uniformly_sample_parameters_within_constraints() / math.pi

    def generate_observation(self):
        state = torch.zeros(self.observation_space.shape)
        state[:6] = self.target_pose.detach().cpu()
        state[6:] = self.current_parameters.detach().cpu()
        return state

    def reset(self):
        self.target_parameters = self.uniformly_sample_parameters_within_constraints()
        force_parameters_within_bounds(self.target_parameters)
        target_transformation = self.open_chain.forward_transformation(
            self.target_parameters
        )
        self.target_pose = transforms.se3_log_map(
            target_transformation.get_matrix()
        )
        # TODO:
        # - Add a level which modulates upwards the noise
        # - Constraint values to the actuator constraints
        # The higher the level is, the higher the noise
        # so that the network learns to solve harder problems
        noise = torch.randn_like(self.target_parameters) * self.max_action * self.max_steps_done
        noise = noise.clamp(-self.max_noise_clip * self.max_steps_done, self.max_noise_clip * self.max_steps_done)
        self.current_parameters = (self.target_parameters.detach().clone() + noise).to(self.device)
        force_parameters_within_bounds(self.current_parameters)
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
        # TODO: use a better threshold
        done = 1 if pose_error < 1e-3 else 0
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
        force_parameters_within_bounds(self.current_parameters)
        # self.current_parameter_index = (self.current_parameter_index + 1) % len(
        #    self.open_chain
        # )
        # print(f"{self.target_parameters} | {self.current_parameters}")
        reward, done = self.compute_reward()
        observation = self.generate_observation()
        if self.current_step >= self.max_steps_done:
            done = 1
        return observation, reward, done

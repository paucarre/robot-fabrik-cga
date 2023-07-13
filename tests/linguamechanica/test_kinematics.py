import unittest
import math
import numpy as np
from linguamechanica.kinematics import (
    DifferentiableOpenChainMechanism,
    UrdfRobotLibrary,
)
import random
import torch


class TestDifferentiableOpenChainMechanism(unittest.TestCase):
    def test_forward_transformation_translation_rotation(self):
        """
        Open Chains:
        - translate 10 meters in z and rotate around x PI rads
        - rotate 90 degrees around x and then translating towards y
          ( which is x wrt the original frame)
        - translate 10 meters in z and rotate around x PI rads

        """
        screws = torch.Tensor(
            [
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        coords = torch.Tensor([[10.0, np.pi], [math.pi / 2.0, 10.0], [10.0, np.pi]])
        initial = torch.Tensor(
            [
                [1, 0, 0, 0],
                [0, math.cos(math.pi / 2.0), -math.sin(math.pi / 2.0), 10.0],
                [0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0), 0.0],
                [0, 0, 0, 1],
            ]
        )
        open_chain = DifferentiableOpenChainMechanism(
            screws, initial, [(0, 100.0), (0, math.pi * 2)]
        )
        matrix = open_chain.forward_transformation(coords)
        expected_matrix = torch.Tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, math.cos(math.pi), -math.sin(math.pi), 0],
                    [0, math.sin(math.pi), math.cos(math.pi), 10.0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 10.0],
                    [0, math.cos(math.pi / 2.0), -math.sin(math.pi / 2.0), 0.0],
                    [0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0), 0.0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 0],
                    [0, math.cos(math.pi), -math.sin(math.pi), 0],
                    [0, math.sin(math.pi), math.cos(math.pi), 10.0],
                    [0, 0, 0, 1],
                ],
            ]
        )
        for i in range(expected_matrix.shape[0]):
            expected_matrix[i, :, :] = expected_matrix[i, :, :] @ initial[:, :]
        self.assertTrue(
            np.isclose(
                expected_matrix.numpy(),
                matrix.numpy(),
                rtol=1e-05,
                atol=1e-05,
            ).all()
        )


class UrdfRobot(unittest.TestCase):
    def test_extract_open_chains(self):
        urdf_robot = UrdfRobotLibrary.dobot_cr5()
        open_chains = urdf_robot.extract_open_chains(0.3)
        for _ in range(100):
            coordinates = []
            for i in range(len(urdf_robot.joint_names)):
                coordinates.append(
                    random.uniform(
                        urdf_robot.joint_limits[i][0], urdf_robot.joint_limits[i][1]
                    )
                )
            coordinates = torch.Tensor(coordinates).unsqueeze(0)
            transformations = urdf_robot.transformations(coordinates)
            for i, transformation in enumerate(transformations):
                computed = open_chains[i].forward_transformation(
                    coordinates[:, : i + 1]
                )
                self.assertTrue(
                    np.isclose(computed, transformation, rtol=1e-05, atol=1e-05).all()
                )

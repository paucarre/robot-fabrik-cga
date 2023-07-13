import unittest
import math
import numpy as np
from fabrik.kinematics import DifferentiableOpenChainMechanism, OpenChainMechanism, UrdfRobotLibrary
import random
import torch

class TestDifferentiableOpenChainMechanism(unittest.TestCase):
    def test_forward_transformation_translation_rotation(self):
        screws = torch.Tensor([[0, 0, 1.0, 0, 0, 0.0], [0.0, 0, 0, 1.0, 0, 0]])
        # translate 10 meters in z and rotate around x PI rads
        coords = torch.Tensor([10.0, np.pi])
        
        screws = torch.Tensor([[0, 0, 0.0, 1.0, 0.0, 0.0], [1.0, 0, 0, 0.0, 0, 0]])
        # rotate 90 degrees around x and then translating towards y ( which is x wrt the original frame)
        coords = torch.Tensor([math.pi / 2.0, 10.0])

        open_chain = DifferentiableOpenChainMechanism(
            screws, torch.eye(4), [(0, 100.0), (0, math.pi * 2)]
        )
        matrix = open_chain.forward_transformation(coords)
        expected_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(math.pi), -math.sin(math.pi), 0],
                [0, math.sin(math.pi), math.cos(math.pi), 10.0],
                [0, 0, 0, 1],
            ]
        )
        print(matrix.squeeze().transpose(0,1))
        self.assertTrue(np.isclose(
                    expected_matrix,
                    matrix.squeeze().transpose(0,1),
                    rtol=1e-05,
                    atol=1e-05,
                ).all())

class TestOpenChainMechanism(unittest.TestCase):
    def test_forward_transformation_translation_rotation(self):
        screws = [np.array([0, 0, 0, 0, 0, 1.0]), np.array([1.0, 0, 0, 0, 0, 0])]
        # translate 10 meters in z and rotate around x PI rads
        coords = [10.0, np.pi]
        open_chain = OpenChainMechanism(
            screws, np.eye(4), [(0, 100.0), (0, math.pi * 2)]
        )
        matrix = open_chain.forward_transformation(coords)
        expected_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(math.pi), -math.sin(math.pi), 0],
                [0, math.sin(math.pi), math.cos(math.pi), 10.0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(expected_matrix, matrix))

    def test_forward_transformation_rotation_translation(self):
        screws = [np.array([1.0, 0, 0, 0, 0, 0.0]), np.array([0.0, 0, 0, 1.0, 0, 0])]
        # rotate 90 degrees around x and then translating towards y ( which is x wrt the original frame)
        coords = [math.pi / 2.0, 10.0]
        open_chain = OpenChainMechanism(
            screws, np.eye(4), [(0, math.pi * 2), (0, 100.0)]
        )
        matrix = open_chain.forward_transformation(coords)
        expected_matrix = np.array(
            [
                [1, 0, 0, 10.0],
                [0, math.cos(math.pi / 2.0), -math.sin(math.pi / 2.0), 0.0],
                [0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0), 0.0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(expected_matrix, matrix))

    def test_forward_transformation_translation_rotation_non_identity_initial(self):
        screws = [np.array([0, 0, 0, 0, 0, 1.0]), np.array([1.0, 0, 0, 0, 0, 0])]
        # translate 10 meters in z and rotate around x PI rads
        coords = [10.0, np.pi]
        initial = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(math.pi / 2.0), -math.sin(math.pi / 2.0), 10.0],
                [0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0), 0.0],
                [0, 0, 0, 1],
            ]
        )
        open_chain = OpenChainMechanism(screws, initial, [(0, 100.0), (0, math.pi * 2)])
        matrix = open_chain.forward_transformation(coords)
        expected_matrix = (
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, math.cos(math.pi), -math.sin(math.pi), 0],
                    [0, math.sin(math.pi), math.cos(math.pi), 10.0],
                    [0, 0, 0, 1],
                ]
            )
            @ initial
        )
        self.assertTrue(np.array_equal(expected_matrix, matrix))


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
            transformations = urdf_robot.transformations(coordinates)
            for i, transformation in enumerate(transformations):
                computed = open_chains[i].forward_transformation(coordinates[: i + 1])
                self.assertTrue(
                    np.isclose(computed, transformation, rtol=1e-05, atol=1e-05).all()
                )

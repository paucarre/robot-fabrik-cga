import unittest
import math
import numpy as np
from linguamechanica.kinematics import (
    DifferentiableOpenChainMechanism,
    UrdfRobotLibrary,
    to_left_multiplied,
)
import random
import torch
from pytorch3d import transforms


class TestDifferentiableOpenChainMechanism(unittest.TestCase):
    def test_inverse_kinematics(self):
        """
        Open Chains:
        - translate 10 meters in z and rotate around x PI rads
        """
        screws = torch.Tensor(
            [
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        initial = torch.eye(4)
        open_chain = DifferentiableOpenChainMechanism(
            screws, initial, [(0, 100.0), (0, math.pi * 2)]
        )
        coords = torch.Tensor([[10.0, np.pi / 4]])
        matrix = open_chain.forward_transformation(coords)
        pose = transforms.se3_log_map(matrix.get_matrix())

        target_pose = pose
        found_coords = open_chain.inverse_kinematics(
            initial_coords=torch.Tensor([[0.0, 0.0]]),
            target_pose=target_pose,
            min_error=1e-2,
            error_weights=torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            parameter_update_rate=torch.Tensor([0.5, 0.5]),
            max_steps=10000,
        )
        assert (found_coords - coords).abs().sum() <= 1e-2

    def test_compute_weighted_error(self):
        error_twist = torch.Tensor(
            [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]
        ).float()
        weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_twist, weights
        )
        (error - torch.Tensor([0.1, 0.4, 0.5])).abs().sum() < 1e-10

    def test_compute_error_twist(self):
        """
        Open Chains:
        - translate 10 meters in z and rotate around x PI rads
        """
        screws = torch.Tensor(
            [
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        initial = torch.eye(4)
        open_chain = DifferentiableOpenChainMechanism(
            screws, initial, [(0, 100.0), (0, math.pi * 2)]
        )
        target_pose = torch.Tensor([[0, 0, 0, 0, 0, 0]])
        # test zero pose and zero coords
        coords = torch.Tensor([[0.0, 0.0]])
        error_twist = open_chain.compute_error_twist(coords, target_pose)
        assert error_twist.abs().sum() < 1e-10
        # test movement of 10 from identity target
        coords = torch.Tensor([[10.0, 0.0]])
        error_twist = open_chain.compute_error_twist(coords, target_pose)
        assert (
            error_twist - torch.Tensor([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
        ).abs().sum() < 1e-10
        # test rotation of 45 deg. from identity
        coords = torch.Tensor([[0.0, np.pi / 4]])
        error_twist = open_chain.compute_error_twist(coords, target_pose)
        assert (
            error_twist - torch.Tensor([[0.0, 0.0, 0.0, np.pi / 4, 0.0, 0.0]])
        ).abs().sum() < 1e-10

    def test_jacobian(self):
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
        jacobian = open_chain.jacobian(coords)
        """
        Verify size is [ Batch, Twist, Coordinate]
        """
        assert jacobian.shape == torch.Size([3, 6, 2])
        """
        Translation parametes do *not* affect rotation velocities
        """
        rotation_idx = [3, 4, 5]
        translation_idx = [0, 1, 2]
        translation_coords = (
            torch.Tensor([[1, 0], [0, 1], [1, 0]]).unsqueeze(1).expand(jacobian.shape)
        )
        rotation_jacobian_by_translation_coords = jacobian[:, rotation_idx, :][
            translation_coords[:, rotation_idx, :] == 1
        ]
        assert rotation_jacobian_by_translation_coords.abs().sum() < 1e-10
        """
        Rotation parameters affect rotation velocities.
        """
        rotation_coords = 1 - translation_coords
        rotation_jacobian_by_rotation_coords = jacobian[:, rotation_idx, :][
            rotation_coords[:, rotation_idx, :] == 1
        ]
        assert rotation_jacobian_by_rotation_coords.abs().sum() > 0.0
        """
        Rotation parameters affect translation velocities
          - When rotation happens, translation can (and often does) take place.
          (e.g. robotic arms move using rotation parameters to move the robot)
          - SO(3) is a *semi* product: translation parameters do not
          affect angular velocities but rotation parameters do affect
          translation.
        """
        translation_jacobian_by_rotation_coords = jacobian[:, translation_idx, :][
            rotation_coords[:, translation_idx, :] == 1
        ]
        assert translation_jacobian_by_rotation_coords.abs().sum() > 0.0
        """
        Translation parameters affect translation velocities
        """
        translation_jacobian_by_translation_coords = jacobian[:, translation_idx, :][
            translation_coords[:, translation_idx, :] == 1
        ]
        assert translation_jacobian_by_translation_coords.abs().sum() > 0.0
        jacobian_pseudoinverse = torch.linalg.pinv(jacobian)
        velocity_delta = torch.ones([3, 6, 1]) * 0.01
        parameter_delta = torch.bmm(jacobian_pseudoinverse, velocity_delta)
        print(parameter_delta)

    def test_forward_transformation(self):
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
                to_left_multiplied(expected_matrix),
                matrix.get_matrix(),
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
                computed = (
                    open_chains[i]
                    .forward_transformation(coordinates[:, : i + 1])
                    .get_matrix()
                )
                self.assertTrue(
                    np.isclose(
                        computed.squeeze(),
                        to_left_multiplied(torch.Tensor(transformation)),
                        rtol=1e-05,
                        atol=1e-05,
                    ).all()
                )

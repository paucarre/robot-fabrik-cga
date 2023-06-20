import unittest
import math
import numpy as np
from fabrik.kinematics import OpenChainMechanism

class TestOpenChainMechanism(unittest.TestCase):

    def test_forward_transformation_translation_rotation(self):
        screws = [np.array([0, 0, 0, 0, 0, 1.0]), np.array([1.0, 0, 0, 0, 0, 0])]
        # translate 10 meters in z and rotate around x PI rads
        coords = [10.0, np.pi]
        open_chain = OpenChainMechanism(screws, np.eye(4))
        matrix = open_chain.forward_transformation(coords)
        expected_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(math.pi), -math.sin(math.pi), 0],
                [0, math.sin(math.pi),  math.cos(math.pi), 10.0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(expected_matrix, matrix))


    def test_forward_transformation_rotation_translation(self):
        screws = [np.array([1.0, 0, 0, 0, 0, 0.0]), np.array([0.0, 0, 0, 1.0, 0, 0])]
        # rotate 90 degrees around x and then translating towards y ( which is x wrt the original frame)
        coords = [math.pi / 2.0, 10.0]
        open_chain = OpenChainMechanism(screws, np.eye(4))
        matrix = open_chain.forward_transformation(coords)
        expected_matrix = np.array(
            [
                [1, 0, 0, 10.0],
                [0, math.cos(math.pi / 2.0), -math.sin(math.pi / 2.0), 0.0],
                [0, math.sin(math.pi / 2.0),  math.cos(math.pi / 2.0), 0.0],
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
                [0, math.cos(math.pi / 2.0), -math.sin(math.pi/ 2.0), 10.0],
                [0, math.sin(math.pi / 2.0),  math.cos(math.pi/ 2.0), 0.0],
                [0, 0, 0, 1],
            ]
        )
        open_chain = OpenChainMechanism(screws, initial)
        matrix = open_chain.forward_transformation(coords)
        expected_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(math.pi), -math.sin(math.pi), 0],
                [0, math.sin(math.pi),  math.cos(math.pi), 10.0],
                [0, 0, 0, 1],
            ]
        ) @ initial
        self.assertTrue(np.array_equal(expected_matrix, matrix))

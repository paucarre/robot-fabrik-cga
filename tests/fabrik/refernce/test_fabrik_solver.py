import unittest
from fabrik.cga import ConformalGeometricAlgebra
from fabrik.reference.fabrik_solver import FabrikSolver
from fabrik.reference.point_chain import PointChain
from fabrik.reference.joint_chain import JointChain
from fabrik.reference.joint import Joint
import math

fabrik_solver = FabrikSolver()
cga = ConformalGeometricAlgebra(1e-11)


class TestFabrikSolver(unittest.TestCase):
    def test_closest_point_to_pair_of_points_in_line_intersecting_with_sphere(self):
        sphere = cga.sphere(cga.point(1.0, 1.0, 1.0), math.sqrt(3.0))
        line = cga.line(cga.point(0.0, 0.0, 0.0), cga.point(1.0, 1.0, 1.0))
        point = cga.point(-1.0, -1.0, -1.0)
        closest_point = fabrik_solver.closest_point_to_pair_of_points_in_line_intersecting_with_sphere(
            point, line, sphere
        )
        self.assertEqual(closest_point, cga.point(0, 0, 0))

    def test_to_rotors(self):
        points = [
            cga.point(0, 0, 0),
            cga.point(0, 1, 0),
            cga.point(0, 2, 0),
            cga.point(1, 2, 0),
        ]
        point_chain = PointChain(points, cga)
        rotors = fabrik_solver.to_rotors(point_chain)
        expected_angles = [math.pi / 2.0, 0.0, -math.pi / 2.0]
        angles = [cga.angle_from_rotor(rotor) for rotor in rotors]
        for expected_angle, angle in list(zip(expected_angles, angles)):
            self.assertTrue(abs(expected_angle - angle) < 1e-6)

    def test_solve_2_joints_constrained(self):
        first_joint = Joint(math.pi / 2.0, 20.0)
        second_joint = Joint(math.pi / 2.0, 100.0)
        joint_chain = JointChain([first_joint, second_joint])
        target_point = cga.sandwich(
            cga.point(120.0, 0.0, 0.0), cga.rotor(cga.e1 ^ cga.e2, math.pi / 4.0)
        )
        fabrik_solver = FabrikSolver()
        positions = fabrik_solver.solve_old(joint_chain, target_point)
        rotors = fabrik_solver.to_rotors(positions)
        angles = [cga.angle_from_rotor(rotor) for rotor in rotors]
        for angle in angles:
            self.assertTrue(angle < math.pi / 4.0)
        self.assertTrue(cga.distance(positions.last(), target_point) < 0.2)

    def test_solve_2_joints(self):
        first_joint = Joint(math.pi, 20.0)
        second_joint = Joint(math.pi, 100.0)
        joint_chain = JointChain([first_joint, second_joint])
        target_point = cga.point(0.0, 120.0, 0.0)
        fabrik_solver = FabrikSolver()
        positions = fabrik_solver.solve(joint_chain, target_point)
        self.assertTrue(cga.distance(positions.last(), target_point) < 0.2)

    def test_solve_3_joints(self):
        first_joint = Joint(2.0 * math.pi, 20.0)
        second_joint = Joint(2.0 * math.pi, 100.0)
        third_joint = Joint(2.0 * math.pi, 100.0)
        joint_chain = JointChain([first_joint, second_joint, third_joint])
        target_point = cga.sandwich(
            cga.point(120.0, 0.0, 0.0), cga.rotor(cga.e1 ^ cga.e2, math.pi / 4.0)
        )
        fabrik_solver = FabrikSolver()
        positions = fabrik_solver.solve(joint_chain, target_point)
        self.assertTrue(cga.distance(positions[-1], target_point) < 1e-9)

    def test_solve_3_joints_unreacheable(self):
        first_joint = Joint(2.0 * math.pi, 20.0)
        second_joint = Joint(2.0 * math.pi, 100.0)
        third_joint = Joint(2.0 * math.pi, 100.0)
        joint_chain = JointChain([first_joint, second_joint, third_joint])
        target_point = cga.sandwich(
            cga.point(420.0, 0.0, 0.0), cga.rotor(cga.e1 ^ cga.e2, math.pi / 4.0)
        )
        fabrik_solver = FabrikSolver()
        positions = fabrik_solver.solve(joint_chain, target_point)
        expected_target_point = cga.sandwich(
            cga.point(220.0, 0.0, 0.0), cga.rotor(cga.e1 ^ cga.e2, math.pi / 4.0)
        )
        self.assertTrue(cga.distance(positions[-1], expected_target_point) < 1e-9)

    def test_solve_3_joints_orientation(self):
        first_joint = Joint(2.0 * math.pi, 50.0)
        second_joint = Joint(2.0 * math.pi, 50.0)
        third_joint = Joint(2.0 * math.pi, 50.0)
        joint_chain = JointChain([first_joint, second_joint, third_joint])
        target_point = cga.sandwich(
            cga.point(50.0, 0.0, 0.0), cga.rotor(cga.e1 ^ cga.e2, math.pi / 4.0)
        )
        fabrik_solver = FabrikSolver()
        target_orientation = cga.e1 + cga.e2 + cga.e3
        positions = fabrik_solver.solve(joint_chain, target_point, target_orientation)
        self.assertTrue(cga.distance(positions[-1], target_point) < 1e-9)
        orientation = cga.to_vector(positions[-1]) - cga.to_vector(positions[-2])
        self.assertTrue(
            cga.vector_norm(
                cga.normalize_vector(orientation)
                - cga.normalize_vector(target_orientation)
            )
            < 1e-9
        )
        self.assertTrue(
            math.sqrt(cga.vector_norm(orientation)) - joint_chain[-1].distance < 1e-9
        )

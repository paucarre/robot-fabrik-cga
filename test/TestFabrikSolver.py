import unittest
from fabrik.ConformalGeometricAlgebra import ConformalGeometricAlgebra
from clifford import *
from fabrik.FabrikSolver import FabrikSolver
from fabrik.PointChain import PointChain
from fabrik.JointChain import JointChain, Joint
import math
fabrik_solver = FabrikSolver()
cga = ConformalGeometricAlgebra(1e-11)

class TestFabrikSolver(unittest.TestCase):

    def test_closestPointToPairOfPointsInLineIntersectingWithSphere(self):
      sphere = cga.sphere(cga.point(1.0, 1.0, 1.0), math.sqrt(3.0))
      line = cga.line(cga.point(0.0,0.0,0.0), cga.point(1.0, 1.0, 1.0))
      point = cga.point(-1.0,-1.0,-1.0)
      closest_point = fabrik_solver.closestPointToPairOfPointsInLineIntersectingWithSphere(point, line, sphere)
      self.assertEqual(closest_point, cga.point(0,0,0))

    def test_toRotors(self):
        points = [cga.point(0,0,0), cga.point(0,1,0), cga.point(0,2,0), cga.point(1,2,0)]
        point_chain = PointChain(points, cga)
        rotors = fabrik_solver.toRotors(point_chain)
        expected_angles = [math.pi / 2.0, 0.0, -math.pi / 2.0]
        angles = [cga.angleFromRotor(rotor) for rotor in rotors]
        for (expected_angle, angle) in list(zip(expected_angles,angles)):
            self.assertTrue(abs(expected_angle - angle) < 1e-6)

    def test_solve_2_joints(self):
        first_joint = Joint(math.pi, 20.0)
        second_joint = Joint(math.pi, 100.0)
        joint_chain = JointChain([first_joint, second_joint])
        target_point = cga.point(0.0, 120.0, 0.0)
        fabrik_solver = FabrikSolver()
        positions = fabrik_solver.solve(joint_chain, target_point)
        self.assertTrue(cga.distance(positions.last(), target_point) < 0.2)

    def test_solve_2_joints_constrained(self):
        first_joint = Joint(math.pi / 2.0, 20.0)
        second_joint = Joint(math.pi / 2.0 , 100.0)
        joint_chain = JointChain([first_joint, second_joint])
        target_point = cga.sandwich(cga.point(120.0, 0.0, 0.0), cga.rotation(cga.e1^cga.e2, math.pi /4.0))
        fabrik_solver = FabrikSolver()
        positions = fabrik_solver.solve(joint_chain, target_point)
        rotors = fabrik_solver.toRotors(positions)
        angles = [cga.angleFromRotor(rotor) for rotor in rotors]
        for angle in angles:
            self.assertTrue(angle < math.pi / 4.0)
        self.assertTrue(cga.distance(positions.last(), target_point) < 0.2)

    def test_solve_3_joints(self):
        first_joint  = Joint(2.0 * math.pi, 20.0)
        second_joint = Joint(2.0 * math.pi, 100.0)
        third_joint  = Joint(2.0 * math.pi, 100.0)
        joint_chain  = JointChain([first_joint, second_joint, third_joint])
        target_point = cga.sandwich(cga.point(120.0, 0.0, 0.0), cga.rotation(cga.e1^cga.e2, math.pi /4.0))
        fabrik_solver = FabrikSolver()
        positions = fabrik_solver.solve(joint_chain, target_point)
        self.assertTrue(cga.distance(positions.last(), target_point) < 1e-9)

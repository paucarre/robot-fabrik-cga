import unittest
from ConformalGeometricAlgebra import ConformalGeometricAlgebra
from clifford import *
from FabrikSolver import FabrikSolver
from PointChain import PointChain

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

import unittest
from ConformalGeometricAlgebra import ConformalGeometricAlgebra
from clifford import *
from FabrikSolver import FabrikSolver

fabrik_solver = FabrikSolver()
cga = ConformalGeometricAlgebra(1e-11)

class TestFabrikSolver(unittest.TestCase):

    def test_closestPointToPairOfPointsInLineIntersectingWithSphere(self):
      sphere = cga.sphere(cga.point(1.0, 1.0, 1.0), math.sqrt(3.0))
      line = cga.line(cga.point(0.0,0.0,0.0), cga.point(1.0, 1.0, 1.0))
      point = cga.point(-1.0,-1.0,-1.0)
      closest_point = fabrik_solver.closestPointToPairOfPointsInLineIntersectingWithSphere(point, line, sphere)
      self.assertEqual(closest_point, cga.point(0,0,0))

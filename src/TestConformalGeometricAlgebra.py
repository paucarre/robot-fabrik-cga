import unittest
from ConformalGeometricAlgebra import ConformalGeometricAlgebra
cga = ConformalGeometricAlgebra()
from clifford import *

class TestConformalGeometricAlgebra(unittest.TestCase):

    def test_homogeneousPoint(self):
      point = cga.point(1, -2, 3)
      self.assertNotEqual(- point * cga.e_inf, 1.0)
      homogenous_point = cga.homogeneousPoint(point)
      self.assertEqual(- homogenous_point | cga.e_inf, 1.0)

    def test_toVector(self):
      point = cga.point(1, -2, 3)
      vector = cga.toVector(point)
      self.assertEqual(vector, cga.vector(1, -2, 3))

    def test_translation(self):
      point = cga.point(1, -2, 3)
      translation = cga.translation(cga.vector(1, -4, 2))
      moved_point = cga.sandwich(point, translation)
      self.assertEqual(moved_point, cga.point(2, -6, 5))

    def test_project(self):
      first_point = cga.point(1, -2, 3)
      second_point = cga.point(4, 1, -8)
      point_pair = second_point ^ first_point
      projected_first_point, projected_second_point = cga.project(point_pair)
      self.assertEqual(cga.homogeneousPoint(projected_first_point), first_point)
      self.assertEqual(cga.homogeneousPoint(projected_second_point), second_point)

    def test_line(self):
      line_horitzontal = cga.line(cga.point(1.0, 0.0, 0.0), cga.point(-1.0, 0.0, 0.0))
      line_vertical = cga.line(cga.point(0.0, 1.0, 0.0), cga.point(0.0, -1.0, 0.0))
      origin = cga.homogeneousPoint(line_horitzontal.meet(line_vertical) | cga.e_inf)
      self.assertEqual(origin, cga.e_origin)

    def test_sphere(self):
      sphere = cga.sphere(cga.point(1.0, 1.0, 1.0), 10)
      line = cga.line(cga.e_origin, cga.point(1.0, 0.0, 0.0))
      projected_first_point, projected_second_point = cga.project(line.meet(sphere))
      self.assertTrue(math.sqrt(abs(projected_first_point | cga.point(-9, 0, 0))) < 1e-2)
      self.assertTrue(math.sqrt(abs(projected_second_point | cga.point(11, 0, 0))) < 1e-2)

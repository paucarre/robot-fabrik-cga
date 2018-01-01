import unittest
from ConformalGeometricAlgebra import ConformalGeometricAlgebra
from clifford import *

cga = ConformalGeometricAlgebra(1e-20)

class TestConformalGeometricAlgebra(unittest.TestCase):

    def test_toRotor_point(self):
      source_point = cga.point(1.0, -10.0, 4.0)
      rotation_plane =  cga.e2 ^ cga.e3
      expected_rotor = cga.rotation(rotation_plane, math.pi)
      expected_destination_point = cga.sandwich(source_point, expected_rotor)
      computed_rotor = cga.toRotor(cga.toVector(source_point), cga.toVector(expected_destination_point))
      computed_destination_point = cga.sandwich(source_point, computed_rotor)
      self.assertEqual(cga.homogeneousPoint(computed_destination_point), cga.homogeneousPoint(expected_destination_point))

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

    def test_normalizeVector(self):
      vector = cga.vector(-4, 1, -8)
      normalized_vector = cga.normalizeVector(vector)
      self.assertEqual(normalized_vector * ~normalized_vector, 1.0)
      projection = vector | normalized_vector
      self.assertEqual(vector / projection, normalized_vector)

    def test_direction(self):
      source_position = cga.point(4, 1, -8)
      destination_position = cga.point(5, -2, -9)
      direction = cga.direction(source_position, destination_position)
      self.assertEqual(direction, cga.normalizeVector(cga.vector(1, -3, -1)))

    def test_angle(self):
      first_vector = cga.vector(1, 1, 0)
      second_vector = cga.vector(0, 0, 1)
      angle = cga.angle(first_vector, second_vector)
      self.assertTrue(abs(angle - (math.pi / 2.0)) < 1e-10)

    def test_distance(self):
      first_vector = cga.point(math.cos(math.pi/2), math.sin(math.pi/2), 0)
      second_vector = cga.point(0, 0, 1)
      distance = cga.distance(first_vector, second_vector)
      self.assertTrue(abs(distance - math.sqrt(2)) < 1e-10)

    def test_rotation(self):
      point = cga.point(math.cos(math.pi/2), math.sin(math.pi/2), 0)
      rotation = cga.rotation(cga.e1 ^ cga.e3, math.pi / 2.0)
      rotated_point = cga.sandwich(point, rotation)
      self.assertEqual(rotated_point, cga.point(0, math.sin(math.pi/2), math.cos(math.pi/2)))

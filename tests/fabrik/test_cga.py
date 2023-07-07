import unittest
from fabrik.cga import ConformalGeometricAlgebra
import math

cga = ConformalGeometricAlgebra(1e-20)


class TestConformalGeometricAlgebra(unittest.TestCase):
    def test_to_rotor_point_1(self):
        source_point = cga.point(1.0, -10.0, 4.0)
        rotation_plane = cga.e2 ^ cga.e3
        expected_rotor = cga.rotor(rotation_plane, math.pi)
        expected_destination_point = cga.sandwich(source_point, expected_rotor)
        computed_rotor = cga.to_rotor(
            cga.to_vector(source_point), cga.to_vector(expected_destination_point)
        )
        computed_destination_point = cga.sandwich(source_point, computed_rotor)
        self.assertEqual(
            cga.homogeneous_point(computed_destination_point),
            cga.homogeneous_point(expected_destination_point),
        )

    def test_to_rotor_point_2(self):
        source_point = cga.point(-2.0, 1.0, -10.0)
        rotation_plane = cga.e2 ^ cga.e3
        expected_rotor = cga.rotor(rotation_plane, math.pi)
        expected_destination_point = cga.sandwich(source_point, expected_rotor)
        computed_rotor = cga.to_rotor(
            cga.to_vector(source_point), cga.to_vector(expected_destination_point)
        )
        computed_destination_point = cga.sandwich(source_point, computed_rotor)
        self.assertEqual(
            cga.homogeneous_point(computed_destination_point),
            cga.homogeneous_point(expected_destination_point),
        )

    def test_homogeneous_point(self):
        point = cga.point(1, -2, 3)
        self.assertNotEqual(-point * cga.e_inf, 1.0)
        homogenous_point = cga.homogeneous_point(point)
        self.assertEqual(-homogenous_point | cga.e_inf, 1.0)

    def test_to_vector(self):
        point = cga.point(1, -2, 3)
        vector = cga.to_vector(point)
        self.assertEqual(vector, cga.vector(1, -2, 3))

    def test_translation(self):
        point = cga.point(1, -2, 3)
        translation = cga.translator(cga.vector(1, -4, 2))
        moved_point = cga.sandwich(point, translation)
        self.assertEqual(moved_point, cga.point(2, -6, 5))

    def test_project(self):
        first_point = cga.point(1, -2, 3)
        second_point = cga.point(4, 1, -8)
        point_pair = second_point ^ first_point
        projected_first_point, projected_second_point = cga.project(point_pair)
        self.assertEqual(cga.homogeneous_point(projected_first_point), first_point)
        self.assertEqual(cga.homogeneous_point(projected_second_point), second_point)

    def test_line(self):
        line_horitzontal = cga.line(cga.point(1.0, 0.0, 0.0), cga.point(-1.0, 0.0, 0.0))
        line_vertical = cga.line(cga.point(0.0, 1.0, 0.0), cga.point(0.0, -1.0, 0.0))
        origin = cga.homogeneous_point(line_horitzontal.meet(line_vertical) | cga.e_inf)
        self.assertEqual(origin, cga.e_origin)

    def test_sphere(self):
        sphere = cga.sphere(cga.point(1.0, 1.0, 1.0), 10)
        line = cga.line(cga.e_origin, cga.point(1.0, 0.0, 0.0))
        projected_first_point, projected_second_point = cga.project(line.meet(sphere))
        self.assertTrue(
            math.sqrt(abs(projected_first_point | cga.point(-9, 0, 0))) < 1e-2
        )
        self.assertTrue(
            math.sqrt(abs(projected_second_point | cga.point(11, 0, 0))) < 1e-2
        )

    def test_normalize_vector(self):
        vector = cga.vector(-4, 1, -8)
        normalized_vector = cga.normalize_vector(vector)
        self.assertEqual(normalized_vector * ~normalized_vector, 1.0)
        projection = vector | normalized_vector
        self.assertEqual(vector / projection, normalized_vector)

    def test_direction(self):
        source_position = cga.point(4, 1, -8)
        destination_position = cga.point(5, -2, -9)
        direction = cga.direction(source_position, destination_position)
        self.assertEqual(direction, cga.normalize_vector(cga.vector(1, -3, -1)))

    def test_angle(self):
        first_vector = cga.vector(1, 1, 0)
        second_vector = cga.vector(0, 0, 1)
        angle = cga.angle(first_vector, second_vector)
        self.assertTrue(abs(angle - (math.pi / 2.0)) < 1e-10)

    def test_distance(self):
        first_vector = cga.point(math.cos(math.pi / 2), math.sin(math.pi / 2), 0)
        second_vector = cga.point(0, 0, 1)
        distance = cga.distance(first_vector, second_vector)
        self.assertTrue(abs(distance - math.sqrt(2)) < 1e-10)

    def test_rotation(self):
        point = cga.point(math.cos(math.pi / 2), math.sin(math.pi / 2), 0)
        rotation = cga.rotor(cga.e1 ^ cga.e3, math.pi / 2.0)
        rotated_point = cga.sandwich(point, rotation)
        self.assertEqual(
            rotated_point, cga.point(0, math.sin(math.pi / 2), math.cos(math.pi / 2))
        )

    def test_point_distance(self):
        point_a = cga.point(0.0, 0.0, 0.0)
        point_b = cga.point(0.0, 1.0, 0.0)
        distance = cga.point_distance(point_a, point_b)
        self.assertTrue(abs(distance - 1.0) < 1e-10)
        point_a = cga.point(0.0, 1.0, 0.0)
        point_b = cga.point(0.0, 1.0, 2.0)
        distance = cga.point_distance(point_a, point_b)
        self.assertTrue(abs(distance - 2.0) < 1e-10)
        point_a = cga.point(1.0, 1.0, 0.0)
        point_b = cga.point(0.0, 1.0, 2.0)
        distance = cga.point_distance(point_a, point_b)
        self.assertTrue(abs(distance - math.sqrt(5.0)) < 1e-10)

    def test_plane_from_non_colinear_points(self):
        point_1 = cga.point(0.0, 0.0, 1.0)
        point_2 = cga.point(1.0, 0.0, 1.0)
        point_3 = cga.point(0.0, 1.0, 1.1)
        plane = cga.plane_from_non_colinear_points(point_1, point_2, point_3)
        expected_normal = cga.vector(0.0, -0.1, 1.0)
        self.assertTrue(
            abs(cga.vector_norm(expected_normal - cga.normal_from_plane(plane))) < 0.01
        )
        point_1 = cga.point(0.0, 0.0, 1.0)
        point_2 = cga.point(1.0, 0.0, 1.1)
        point_3 = cga.point(0.0, 1.0, 1.0)
        plane = cga.plane_from_non_colinear_points(point_1, point_2, point_3)
        expected_normal = cga.vector(-0.1, 0.0, 1.0)
        self.assertTrue(
            abs(cga.vector_norm(expected_normal - cga.normal_from_plane(plane))) < 0.01
        )
        point_1 = cga.point(0.0, 0.0, 1.1)
        point_2 = cga.point(1.0, 0.0, 1.0)
        point_3 = cga.point(0.0, 1.0, 1.0)
        plane = cga.plane_from_non_colinear_points(point_1, point_2, point_3)
        expected_normal = cga.vector(0.1, 0.1, 1.0)
        self.assertTrue(
            abs(cga.vector_norm(expected_normal - cga.normal_from_plane(plane))) < 0.01
        )


"""
    def test_distance_between_line_and_point(self):
      pointA = cga.point(1.0, 0.0, 0.0)
      pointB = cga.point(0.0, 1.0, 0.0)
      point_target = cga.point(0.0, 0.0, 1.0)
      distance = cga.distance_between_line_and_point(pointA, pointB, point_target)
      self.assertTrue(abs(distance - 0.5) < 1e-10)
"""

from clifford import Cl, eps, pretty
import math


class ConformalGeometricAlgebra(object):
    def __init__(self, resolution=1e-15):
        self.layout, self.blades = Cl(4, 1)
        pretty()
        self.resolution = resolution
        eps(self.resolution)
        self.e1, self.e2, self.e3, self.e_hat, self.e = [
            self.blades["e%i" % k] for k in range(1, 6)
        ]
        self.e_origin = 0.5 ^ (self.e - self.e_hat)
        self.e_inf = self.e_hat + self.e
        self.minkowski_plane = self.e_inf ^ self.e_origin

    def angle_from_rotor(self, rotor):
        cos_half_angle = rotor.lc(1)
        half_angle = math.acos(cos_half_angle)
        angle = half_angle * 2.0
        if (
            float(
                (rotor - cos_half_angle)
                | rotor.lc(
                    (self.e1 ^ self.e2) + (self.e1 ^ self.e3) + (self.e2 ^ self.e3)
                )
            )
            < 0.0
        ):
            angle = -angle
        return angle

    def to_rotor(self, first_vector, second_vector):
        """
        returns a rotor composed of the plane formed
        by the two vectors and the angle between them.
        The first vector should become the second by
        acting on the rotor
        """
        second_vector = self.normalize_vector(second_vector)
        first_vector = self.normalize_vector(first_vector)
        rotation_plane = self.normalize_vector(second_vector ^ first_vector)
        angle = self.angle(first_vector, second_vector)
        return self.rotor(rotation_plane, angle)

    def point_distance(self, point_a, point_b):
        return math.sqrt(-(point_a | point_b) * 2.0)

    def to_point(self, vector):
        return self.homogeneous_point(
            vector + (0.5 ^ ((vector**2) * self.e_inf)) + self.e_origin
        )

    def homogeneous_point(self, point):
        if abs(point | self.e_inf) > 2.0 * self.resolution:
            return point * (-point | self.e_inf).normalInv()
        else:
            # zero point, non-invertible
            return self.point(0.0, 0.0, 0.0)

    def to_vector(self, point):
        return (
            self.homogeneous_point(point) ^ self.minkowski_plane
        ) * self.minkowski_plane

    def rotor(self, bivector, angle):
        return math.cos(angle / 2.0) + (math.sin(angle / 2.0) ^ bivector)

    def translator(self, vector):
        return 1.0 + (0.5 ^ (self.e_inf * vector))

    def project(self, point_pair):
        beta = math.sqrt(abs(point_pair * point_pair))
        point_pair = (1.0 / beta) * point_pair
        projector = 0.5 ^ (1.0 + point_pair)
        first_point = projector * (point_pair | self.e_inf)
        second_point = -~projector * (point_pair | self.e_inf)
        return first_point, second_point

    def sandwich(self, element, transformation):
        return transformation * element * (~transformation)

    def sandwiches(self, element, transformations):
        for transformation in transformations:
            element = self.sandwich(element, transformation)
        return element

    def vector(self, x, y, z):
        return (x ^ self.e1) + (y ^ self.e2) + (z ^ self.e3)

    def point(self, x, y, z):
        return self.to_point(self.vector(x, y, z))

    def line(self, first_point, second_point):
        return (
            self.homogeneous_point(first_point)
            ^ self.homogeneous_point(second_point)
            ^ self.e_inf
        )

    def vector_norm(self, vector):
        return math.sqrt(abs(vector * ~vector))

    def normalize_vector(self, vector):
        norm2 = self.vector_norm(vector)
        if norm2 > self.resolution:
            return vector / norm2
        else:
            return vector

    def direction(self, source_position, destination_position):
        return self.normalize_vector(
            self.to_vector(destination_position) - self.to_vector(source_position)
        )

    def normal_from_plane(self, plane):
        # dual = plane.dual()
        # x = -dual.lc(self.e1)
        # y = -dual.lc(self.e2)
        # z = -dual.lc(self.e3)
        x = -plane.lc(self.e ^ self.e2 ^ self.e3 ^ self.e_hat)
        y = -plane.lc(self.e ^ self.e3 ^ self.e1 ^ self.e_hat)
        z = -plane.lc(self.e ^ self.e1 ^ self.e2 ^ self.e_hat)
        return self.vector(x, y, z)

    def plane_from_two_vectors_and_point(self, vector1, vector2, point):
        point_vector1 = self.sandwich(point, self.translator(vector1))
        point_vector2 = self.sandwich(point, self.translator(vector2))
        return self.plane_from_non_colinear_points(point, point_vector1, point_vector2)

    def dipole_circle_plane_intersection(self, circle, plane):
        '''
        Plane
        https://conformalgeometricalgebra.org/wiki/index.php?title=Plane
        g=
            gxe4235+
            gye4315+
            gze4125+
            gwe3215
        '''
        gx = plane.lc(self.e  ^ self.e2 ^ self.e3 ^ self.e_hat).as_array().sum()
        gy = plane.lc(self.e  ^ self.e3 ^ self.e1 ^ self.e_hat).as_array().sum()
        gz = plane.lc(self.e  ^ self.e1 ^ self.e2 ^ self.e_hat).as_array().sum()
        gw = plane.lc(self.e3 ^ self.e2 ^ self.e1 ^ self.e_hat).as_array().sum()
        '''
        Circle
        https://conformalgeometricalgebra.org/wiki/index.php?title=Circle
        '''
        '''
        cgxe423+
        cgye431+
        cgze412+
        cgwe321+
        '''
        cgx = circle.lc(self.e  ^ self.e2 ^ self.e3).as_array().sum()
        cgy = circle.lc(self.e  ^ self.e3 ^ self.e1).as_array().sum()
        cgz = circle.lc(self.e  ^ self.e1 ^ self.e2).as_array().sum()
        cgw = circle.lc(self.e3 ^ self.e2 ^ self.e1).as_array().sum()
        '''
        cvxe415+
        cvye425+
        cvze435+
        '''
        cvx = circle.lc(self.e ^ self.e1 ^ self.e_hat).as_array().sum()
        cvy = circle.lc(self.e ^ self.e2 ^ self.e_hat).as_array().sum()
        cvz = circle.lc(self.e ^ self.e3 ^ self.e_hat).as_array().sum()
        '''
        cmxe235+
        cmye315+
        cmze125
        '''
        cmx = circle.lc(self.e2 ^ self.e3 ^ self.e_hat).as_array().sum()
        cmy = circle.lc(self.e3 ^ self.e1 ^ self.e_hat).as_array().sum()
        cmz = circle.lc(self.e1 ^ self.e2 ^ self.e_hat).as_array().sum()
        '''
        Clicle - Plane meet
        https://conformalgeometricalgebra.org/wiki/index.php?title=Join_and_meet
        g∨c=
            (gycgz-gzcgy)e41+
            (gzcgx-gxcgz)e42+
            (gxcgy-gycgx)e43+
            (gwcgx-gxcgw)e23+
            (gwcgy-gycgw)e31+
            (gwcgz-gzcgw)e12+
            (gzcmy-gycmz+gwcvx)e15+
            (gxcmz-gzcmx+gwcvy)e25+
            (gycmx-gxcmy+gwcvz)e35-
            (gxcvx+gycvy+gzcvz)e45
        '''
        meet =  ( (gy*cgz-gz*cgy) * (self.e  ^ self.e1) ) + \
                ( (gz*cgx-gx*cgz) * (self.e  ^ self.e2) ) + \
                ( (gx*cgy-gy*cgx) * (self.e  ^ self.e3) ) + \
                ( (gw*cgx-gx*cgw) * (self.e2 ^ self.e3) ) + \
                ( (gw*cgy-gy*cgw) * (self.e3 ^ self.e1) ) + \
                ( (gw*cgz-gz*cgw) * (self.e1 ^ self.e2) ) + \
                ( (gz*cmy-gy*cmz+gw*cvx) * (self.e1 ^ self.e_hat) ) + \
                ( (gx*cmz-gz*cmx+gw*cvy) * (self.e2 ^ self.e_hat) ) + \
                ( (gy*cmx-gx*cmy+gw*cvz) * (self.e3 ^ self.e_hat) ) - \
                ( (gx*cvx+gy*cvy+gz*cvz) * (self.e  ^ self.e_hat) )
        return meet

    """
    def distance_between_line_and_point(self, first_point, second_point, target_point):
        plane = first_point ^ second_point ^ target_point ^ self.e_inf
        print(plane)
        print(~plane)
        #line_direction = self.normalize_vector(self.to_vector(second_point) - self.to_vector(first_point))
        #target_direction = self.normalize_vector(self.to_vector(target_point) - self.to_vector(first_point))
        #plane = self.normalize_vector(line_direction ^ target_direction)
        #normal_line_rotation = self.rotor(plane, math.pi / 2.0)
        #orthogonal_line_direction = self.sandwich(line_direction, normal_line_rotation)


        #line = self.line(first_point, second_point)
        #orthogonal_line = self.line(target_point, self.sandwich(target_point, self.translator(orthogonal_line_direction)))

        #print(line_direction)
        #print(self.to_vector(first_point))
        #print(self.to_vector(second_point))
        #print(self.to_vector(target_point))
        #print(self.to_vector(self.sandwich(target_point, self.translator(orthogonal_line_direction))))
        #print(self.normalize_vector(orthogonal_line))
        #print(self.normalize_vector(line))
        #print(plane)
        #print(orthogonal_line_direction)
        #closest_point_to_line = ~((~orthogonal_line) ^ (~self.normalize_vector(line)))
        #print(closest_point_to_line)
        #print(self.to_vector(closest_point_to_line)) # WRONG
        #print(self.to_vector(target_point)) # OK
        #print(closest_point_to_line | target_point)
        #parallel_line_in_target  = self.normalize_vector(target_point ^ (self.to_vector(second_point) - self.to_vector(first_point)) ^ self.e_inf)
        #return line | parallel_line_in_target
        return 3
    """

    def angle(self, first_vector, second_vector):
        first_vector_normalized = self.normalize_vector(first_vector)
        second_vector_normalized = self.normalize_vector(second_vector)
        cos_angle = float(first_vector_normalized | second_vector_normalized)
        if cos_angle < -1.0:
            cos_angle = -1.0
        if cos_angle > 1.0:
            cos_angle = 1.0
        angle = math.acos(cos_angle)
        # if(angle > math.pi):
        #    angle = (2.0 * math.pi) - angle
        return angle

    def distance(self, origin_point, destination_point):
        distance = destination_point - origin_point
        return math.sqrt(abs(distance * ~distance))

    def sphere(self, center, radius):
        sphere_point_1_translation = self.translator(self.vector(radius, 0.0, 0.0))
        sphere_point_2_translation = self.translator(self.vector(0.0, radius, 0.0))
        sphere_point_3_translation = self.translator(self.vector(0.0, 0.0, radius))
        sphere_point_4_translation = self.translator(self.vector(-radius, 0.0, 0.0))
        sphere = (
            self.sandwich(center, sphere_point_1_translation)
            ^ self.sandwich(center, sphere_point_2_translation)
            ^ self.sandwich(center, sphere_point_3_translation)
            ^ self.sandwich(center, sphere_point_4_translation)
        )
        return sphere

    def circle(self, center, radius):
        circle_point_1_translation = self.translator(self.vector(radius, 0.0, 0.0))
        circle_point_2_translation = self.translator(self.vector(0.0, radius, 0.0))
        circle_point_3_translation = self.translator(self.vector(0.0, 0.0, radius))
        return (
            self.sandwich(center, circle_point_1_translation)
            ^ self.sandwich(center, circle_point_2_translation)
            ^ self.sandwich(center, circle_point_3_translation)
        )

    def circle_from_non_colinear_points(self, point_1, point_2, point_3):
        return point_1 ^ point_2 ^ point_3

    def plane_from_non_colinear_points(self, point_1, point_2, point_3):
        return point_1 ^ point_2 ^ point_3 ^ self.e_inf

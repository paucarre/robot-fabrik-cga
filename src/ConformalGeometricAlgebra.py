from clifford import *
import random
class ConformalGeometricAlgebra(object):

    def __init__(self):
        self.layout, self.blades = Cl(4,1)
        pretty()
        eps(1e-15)
        self.e1, self.e2, self.e3, self.e_hat, self.e = [self.blades['e%i'%k] for k in range(1, 6)]
        self.e_origin = 0.5 ^ (self.e - self.e_hat)
        self.e_inf = self.e_hat + self.e
        self.minkowski_plane = self.e_inf ^ self.e_origin

    def toPoint(self, vector):
        return self.homogeneousPoint(vector + (0.5 ^ ( (vector**2) * self.e_inf ) ) + self.e_origin)

    def homogeneousPoint(self, point):
        if(abs(point | self.e_inf) > 1e-7):
            return point * ( -point | self.e_inf ).normalInv()
        else:
            # zero point, non-invertible
            return self.point(0.0, 0.0, 0.0)

    def toVector(self, point):
        return ( self.homogeneousPoint(point) ^ self.minkowski_plane ) * self.minkowski_plane

    def rotation(self, bivector, angle):
        return math.cos(angle / 2.0) + (math.sin(angle / 2.0) ^ bivector)

    def translation(self, vector):
        return 1.0 + ( 0.5 ^ (self.e_inf * vector) )

    def project(self, point_pair):
        beta = math.sqrt(abs(point_pair * point_pair))
        point_pair = (1.0 / beta) * point_pair
        projector = 0.5 ^ (1.0 + point_pair)
        first_point = projector * (point_pair | self.e_inf)
        second_point = - ~projector * (point_pair | self.e_inf)
        return first_point, second_point

    def sandwich(self, element, transformation):
        return transformation * element * (~transformation)

    def sandwiches(self, element, transformations):
        for transformation in transformations:
            element = self.sandwich(element, transformation)
        return element

    def vector(self, x, y, z):
        return (x^self.e1) + (y^self.e2) + (z^self.e3)

    def point(self, x, y, z):
        return self.toPoint(self.vector(x, y, z))

    def line(self, first_point, second_point):
        return self.homogeneousPoint(first_point) ^ self.homogeneousPoint(second_point) ^ self.e_inf

    def sphere(self, center, radius):
        sphere_point_1_translation = self.translation(self.vector(radius, 0.0, 0.0))
        sphere_point_2_translation = self.translation(self.vector(0.0, radius, 0.0))
        sphere_point_3_translation = self.translation(self.vector(0.0, 0.0, radius))
        sphere_point_4_translation = self.translation(self.vector(-radius, 0.0, 0.0))
        sphere = self.sandwich(center, sphere_point_1_translation) ^ \
                 self.sandwich(center, sphere_point_2_translation) ^ \
                 self.sandwich(center, sphere_point_3_translation) ^ \
                 self.sandwich(center, sphere_point_4_translation)
        return sphere

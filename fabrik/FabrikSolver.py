from fabrik.ConformalGeometricAlgebra import ConformalGeometricAlgebra
from fabrik.PointChain import PointChain
import math

class FabrikSolver(object):

    def __init__(self):
        self.cga = ConformalGeometricAlgebra()
        self.resolution = 1e-10
        self.plane_bivector = self.cga.e1 ^ self.cga.e2

    def closestPointToPairOfPointsInLineIntersectingWithSphere(self, reference_point, line, sphere):
        vector_pair = sphere.meet(line)
        first_point, second_point = self.cga.project(vector_pair)
        if(self.cga.distance(first_point, reference_point) < self.cga.distance(second_point, reference_point)):
            return self.cga.homogeneousPoint(first_point)
        else:
            return self.cga.homogeneousPoint(second_point)

    def getTarget(self, target, forward):
        if(forward):
            return target
        else:
            return self.cga.e_origin

    def error(self, target, point_chain):
        return math.sqrt(abs(target | point_chain.get(0, True))) + math.sqrt(abs(self.cga.e_origin | point_chain.get(0, False)))

    def randomDistortion(self, point):
        random_direction = self.cga.vector(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), 0.0)
        random_translation = self.cga.translator(random_direction)
        random_position = self.cga.sandwich(point, random_translation)
        return random_position

    def resolveAngleConstraints(self, previous_direction, previous_position, current_position, angle, joint):
        max_angle_clockwise_position = self.cga.sandwich(previous_position, self.cga.translator(self.cga.sandwich(joint.distance * previous_direction, self.cga.rotor(self.plane_bivector, angle / 2.0))))
        max_angle_anticlockwise_position = self.cga.sandwich(previous_position,  self.cga.translator(self.cga.sandwich(joint.distance * previous_direction, self.cga.rotor(-self.plane_bivector, angle / 2.0))))
        max_angle_clockwise_distance = self.cga.distance(current_position, max_angle_clockwise_position)
        max_angle_anticlockwise_distance = self.cga.distance(current_position, max_angle_anticlockwise_position)
        if(max_angle_anticlockwise_distance <= max_angle_clockwise_distance):
            return max_angle_anticlockwise_position
        else:
            return max_angle_clockwise_position

    def toRotors(self, point_chain):
        previous_direction = self.cga.vector(1.0, 0.0, 0.0)
        previous_position = None
        rotors = []
        for current_position in point_chain.positions:
            if previous_position is None:
                previous_position = current_position
            else:
                current_direction = self.cga.direction(previous_position, current_position)
                rotor = self.cga.toRotor(previous_direction, current_direction)
                rotors.insert(len(rotors), rotor)
                previous_direction = current_direction
                previous_position = current_position
        return rotors

    def articulationCloseToTarget(self, previous_position, current_position, target_position):
        line = self.cga.line(previous_position, current_position)
        parallel_line_in_target  = self.cga.e_inf ^ target_position ^ (current_position - previous_position)
        #print(f"Line: {line}")
        #print(f"Distance line to itself: {parallel_line_in_target | line}")
        distance = (parallel_line_in_target | line)
        return distance

    def solve(self, joint_chain, target_position, max_iterations=100):
        point_chain = PointChain.fromJoints(joint_chain, self.cga)
        iteration = 0
        forward = True
        while self.error(target_position, point_chain) > self.resolution and iteration < max_iterations:
            previous_direction = None
            if not forward:
                previous_direction  = self.cga.vector(1.0, 0.0, 0.0)
            current_target_position = self.getTarget(target_position, forward)
            point_chain.set(0, forward, current_target_position)
            for index in range(1, len(point_chain)):
                current_position = point_chain.get(index, forward)
                previous_position = point_chain.get(index - 1, forward)
                while(abs(current_position | previous_position) < self.resolution):
                    #print("*** RANDOM DISTORTION APPLIED ***")
                    current_position = self.randomDistortion(current_position)
                distance = self.articulationCloseToTarget(previous_position, current_position, current_target_position)
                #print(f"Distance: {distance}")
                line = self.cga.line(current_position, previous_position)
                joint = joint_chain.get(index - 1, forward)
                sphere = self.cga.sphere(previous_position, joint.distance)
                current_position = self.closestPointToPairOfPointsInLineIntersectingWithSphere(current_position, line, sphere)
                current_direction = self.cga.direction(previous_position, current_position)
                if previous_direction is not None:
                    angle = self.cga.angle(current_direction, previous_direction)
                    if(angle > joint.angle_constraint / 2.0):
                        current_position = self.resolveAngleConstraints(previous_direction, previous_position, current_position, angle, joint)
                        current_direction = self.cga.direction(previous_position, current_position)
                point_chain.set(index, forward, current_position)
                #print(point_chain)
                previous_direction = current_direction
            iteration = iteration + 1
            forward = not forward
        return point_chain

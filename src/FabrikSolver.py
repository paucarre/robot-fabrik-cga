from ConformalGeometricAlgebra import ConformalGeometricAlgebra
from PointChain import PointChain
import math

class FabrikSolver(object):

    def __init__(self):
        self.cga = ConformalGeometricAlgebra()
        self.resolution = 1e-10

    def closestPointToPairOfPointsInLineIntersectingWithSphere(self, reference_point, line, sphere):
        vector_pair = sphere.meet(line)
        first_point, second_point = self.cga.project(vector_pair)
        first_point_distance = math.sqrt(abs(first_point | reference_point))
        second_point_distance = math.sqrt(abs(second_point | reference_point))
        if(first_point_distance < second_point_distance):
            return first_point
        else:
            return second_point

    def getTarget(self, target, forward):
        if(forward):
            return target
        else:
            return self.cga.e_origin

    def error(self, target, point_chain):
        return math.sqrt(abs(target | point_chain.get(0, True))) + math.sqrt(abs(self.cga.e_origin | point_chain.get(0, False)))

    def randomDistortion(self, point):
        random_direction = self.cga.vector(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), 0.0)
        random_translation = self.cga.translation(random_direction)
        random_position = self.cga.sandwich(point, random_translation)
        return random_position

    def direction(self, source_position, destination_position):
        unormalized_direction = self.cga.toVector(destination_position) - self.cga.toVector(source_position)
        normalized_direction = unormalized_direction / math.sqrt(unormalized_direction * ~unormalized_direction)
        return normalized_direction

    def solve(self, joint_chain, target_position, max_iterations=1000):
        point_chain = PointChain(joint_chain, self.cga)
        iteration = 0
        forward = True
        previous_direction = None
        while self.error(target_position, point_chain) > self.resolution and iteration < max_iterations:
            current_target_position = self.getTarget(target_position, forward)
            point_chain.set(0, forward, current_target_position)
            for index in range(1, len(point_chain)):
                current_position = point_chain.get(index, forward)
                previous_position = point_chain.get(index - 1, forward)
                while(abs(current_position | previous_position) < self.resolution):
                    current_position = self.randomDistortion(current_position)
                line = self.cga.line(current_position, previous_position)
                joint = joint_chain.get(index - 1, forward)
                sphere = self.cga.sphere(previous_position, joint.distance)
                current_position = self.closestPointToPairOfPointsInLineIntersectingWithSphere(current_position, line, sphere)
                current_direction = self.direction(previous_position, current_position)
                if index > 1:
                    angle = math.acos(float(current_direction | previous_direction))
                    angle_constraint = joint.angle_constraint
                    if(angle > angle_constraint / 2.0):
                        current_position = self.cga.sandwich(previous_position, self.cga.translation(joint.distance * previous_direction))
                        current_direction = self.direction(previous_position, current_position)
                point_chain.set(index, forward, current_position)
                previous_direction = current_direction
            iteration = iteration + 1
            forward = not forward
        return [self.cga.toVector(position) for position in point_chain.positions ]

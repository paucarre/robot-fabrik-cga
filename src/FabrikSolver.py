from ConformalGeometricAlgebra import ConformalGeometricAlgebra
from PointChain import PointChain
import math

class FabrikSolver(object):

    def __init__(self):
        self.cga = ConformalGeometricAlgebra()
        self.resolution = 1e-10

    def closestPointToPairOfPoinsInLineIntersectingWithSphere(self, reference_point, line, sphere):
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
            return self.cga.point(0.0, 0.0, 0.0)

    def error(self, target, point_chain):
        return math.sqrt(abs(target | point_chain.get(0, True))) + math.sqrt(abs(self.cga.point(0.0, 0.0, 0.0) | point_chain.get(0, False)))

    def randomDistortion(self, point):
        random_direction = self.cga.vector(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), 0.0)
        random_translation = self.cga.translation(random_direction)
        random_position = self.cga.sandwich(point, random_translation)
        return random_position

    def solve(self, joint_chain, target_position, max_iterations=30):
        point_chain = PointChain(joint_chain, self.cga)
        end_effector_position = point_chain.get(0, True)
        iteration = 0
        forward = True
        while self.error(target_position, point_chain) > self.resolution and iteration < max_iterations:
            current_target_position = self.getTarget(target_position, forward)
            # last point goes to target
            point_chain.set(0, forward, current_target_position)
            for index in range(1, len(point_chain)):
                # trace line between target and previous
                while(abs(point_chain.get(index, forward) | point_chain.get(index - 1, forward)) < self.resolution):
                    # they are in the same point and thus cannot form a linev -> apply distortion to a point
                    random_position = self.randomDistortion(point_chain.get(index, forward))
                    point_chain.set(index, forward, random_position)
                line = self.cga.line(point_chain.get(index, forward), point_chain.get(index - 1, forward))
                sphere = self.cga.sphere(point_chain.get(index - 1, forward), joint_chain.get(index - 1, forward).distance)
                closest_point = self.closestPointToPairOfPoinsInLineIntersectingWithSphere(point_chain.get(index, forward), line, sphere)
                point_chain.set(index, forward, closest_point)
            iteration = iteration + 1
            forward = not forward
        return [self.cga.toVector(position) for position in point_chain.positions ]

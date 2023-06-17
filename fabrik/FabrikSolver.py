from fabrik.ConformalGeometricAlgebra import ConformalGeometricAlgebra
from fabrik.PointChain import PointChain
import math


class FabrikSolver(object):
    def __init__(self):
        self.cga = ConformalGeometricAlgebra()
        self.resolution = 1e-10
        self.plane_bivector = self.cga.e1 ^ self.cga.e2

    def closest_point_to_pair_of_points_in_line_intersecting_with_sphere(
        self, reference_point, line, sphere
    ):
        vector_pair = sphere.meet(line)
        first_point, second_point = self.cga.project(vector_pair)
        if self.cga.distance(first_point, reference_point) < self.cga.distance(
            second_point, reference_point
        ):
            return self.cga.homogeneous_point(first_point)
        else:
            return self.cga.homogeneous_point(second_point)

    def get_target(self, target, forward):
        if forward:
            return target
        else:
            return self.cga.e_origin

    def error(self, target, point_chain):
        """
        returns the distance between the two targets and the two origins
        """
        return math.sqrt(abs(target | point_chain.get(0, True))) + math.sqrt(
            abs(self.cga.e_origin | point_chain.get(0, False))
        )

    def random_distortion(self, point):
        random_direction = self.cga.vector(
            random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), 0.0
        )
        random_translation = self.cga.translator(random_direction)
        random_position = self.cga.sandwich(point, random_translation)
        return random_position

    def resolve_angle_constraints(
        self, previous_direction, previous_position, current_position, angle, joint
    ):
        max_angle_clockwise_position = self.cga.sandwich(
            previous_position,
            self.cga.translator(
                self.cga.sandwich(
                    joint.distance * previous_direction,
                    self.cga.rotor(self.plane_bivector, angle / 2.0),
                )
            ),
        )
        max_angle_anticlockwise_position = self.cga.sandwich(
            previous_position,
            self.cga.translator(
                self.cga.sandwich(
                    joint.distance * previous_direction,
                    self.cga.rotor(-self.plane_bivector, angle / 2.0),
                )
            ),
        )
        max_angle_clockwise_distance = self.cga.distance(
            current_position, max_angle_clockwise_position
        )
        max_angle_anticlockwise_distance = self.cga.distance(
            current_position, max_angle_anticlockwise_position
        )
        if max_angle_anticlockwise_distance <= max_angle_clockwise_distance:
            return max_angle_anticlockwise_position
        else:
            return max_angle_clockwise_position

    def to_rotors(self, point_chain, initial_direction=(1.0, 0.0, 0.0)):
        """
        returns rotors that conform the path of points
        """
        previous_direction = self.cga.vector(
            initial_direction[0], initial_direction[1], initial_direction[2]
        )
        previous_position = None
        rotors = []
        for current_position in point_chain.positions:
            if previous_position is None:
                previous_position = current_position
            else:
                current_direction = self.cga.direction(
                    previous_position, current_position
                )
                rotor = self.cga.to_rotor(previous_direction, current_direction)
                rotors.insert(len(rotors), rotor)
                previous_direction = current_direction
                previous_position = current_position
        return rotors

    def articulation_close_to_target(
        self, previous_position, current_position, target_position
    ):
        line = self.cga.line(previous_position, current_position)
        parallel_line_in_target = (
            self.cga.e_inf ^ target_position ^ (current_position - previous_position)
        )
        # print(f"Line: {line}")
        # print(f"Distance line to itself: {parallel_line_in_target | line}")
        distance = parallel_line_in_target | line
        return distance

    def fabrik_solver(self, joint_chain, target_position, max_iterations=100):
        """
        Input: The joint positions pi for i = 1,...,n, the
        target position t and the distances between each
        joint  di = jpi+1  pij for i = 1,...,n  1.
        Output: The new joint positions pi for i = 1,...,n.
        """
        point_chain = PointChain.from_joints(joint_chain, self.cga)
        root_to_target_distance = self.cga.point_distance(target_position, self.e_origin)
        
        """
        1.1 % The distance between root and target
        1.2 dist = jp1  tj
        1.3 % Check whether the target is within reach
        1.4 if dist > d1 + d2 +...+ dn1 then
        1.5 % The target is unreachable
        1.6 for i = 1,...,n  1 do
        1.7 % Find the distance ri between the target t and
        the joint
        position pi
        1.8 ri = jt  pij
        1.9 ki = di/ri
        1.10 % Find the new joint positions pi.
        1.11 pi+1 = (1  ki) pi + kit
        1.12 end
        1.13 else
        1.14 % The target is reachable; thus, set as b the
        initial position of the
        joint p1
        1.15 b = p1
        1.16 % Check whether the distance between the end
        effector pn
        and the target t is greater than a tolerance.
        1.17 difA = jpn  tj
        1.18 while difA > tol do
        1.19 % STAGE 1: FORWARD REACHING
        1.20 % Set the end effector pn as target t
        1.21 pn = t
        1.22 for i = n  1,...,1 do
        1.23 % Find the distance ri between the new joint
        position
        pi+1 and the joint pi
        1.24 ri = jpi+1  pij
        1.25 ki = di/ri
        1.26 % Find the new joint positions pi.
        1.27 pi = (1  ki) pi+1 + kipi
        1.28 end
        1.29 % STAGE 2: BACKWARD REACHING
        1.30 % Set the root p1 its initial position.
        1.31 p1 = b
        1.32 for i = 1,...,n  1 do
        1.33 % Find the distance ri between the new joint
        position pi
        and the joint pi+1
        1.34 ri = jpi+1  pij
        1.35 ki = di/ri
        1.36 % Find the new joint positions pi.
        1.37 pi+1 = (1  ki)pi + kipi+1
        1.38 end
        1.39 difA = jpn  tj
        1.40 end
        1.41 end
        """

    def solve(self, joint_chain, target_position, max_iterations=100):
        point_chain = PointChain.from_joints(joint_chain, self.cga)
        iteration = 0
        forward = True
        while (
            self.error(target_position, point_chain) > self.resolution
            and iteration < max_iterations
        ):
            previous_direction = None
            if not forward:
                previous_direction = self.cga.vector(1.0, 0.0, 0.0)
            current_target_position = self.get_target(target_position, forward)
            point_chain.set(0, forward, current_target_position)
            for index in range(1, len(point_chain)):
                current_position = point_chain.get(index, forward)
                previous_position = point_chain.get(index - 1, forward)
                while abs(current_position | previous_position) < self.resolution:
                    # print("*** RANDOM DISTORTION APPLIED ***")
                    current_position = self.random_distortion(current_position)
                distance = self.articulation_close_to_target(
                    previous_position, current_position, current_target_position
                )
                # print(f"Distance: {distance}")
                line = self.cga.line(current_position, previous_position)
                joint = joint_chain.get(index - 1, forward)
                sphere = self.cga.sphere(previous_position, joint.distance)
                current_position = self.closest_point_to_pair_of_points_in_line_intersecting_with_sphere(
                    current_position, line, sphere
                )
                current_direction = self.cga.direction(
                    previous_position, current_position
                )
                if previous_direction is not None:
                    angle = self.cga.angle(current_direction, previous_direction)
                    if angle > joint.angle_constraint / 2.0:
                        current_position = self.resolve_angle_constraints(
                            previous_direction,
                            previous_position,
                            current_position,
                            angle,
                            joint,
                        )
                        current_direction = self.cga.direction(
                            previous_position, current_position
                        )
                point_chain.set(index, forward, current_position)
                # print(point_chain)
                previous_direction = current_direction
            iteration = iteration + 1
            forward = not forward
        return point_chain

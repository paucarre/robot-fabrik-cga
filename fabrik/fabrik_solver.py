from fabrik.cga import ConformalGeometricAlgebra
from fabrik.point_chain import PointChain
import math
import logging


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

    def solve(self, joint_chain, target_position, tolearance=1e-6, max_iterations=100):
        """
        Input: The joint positions pi for i = 1,...,n, the
        target position t and the distances between each
        joint  di = jpi+1  pij for i = 1,...,n  1.
        Output: The new joint positions pi for i = 1,...,n.
        """
        point_chain = PointChain.from_joints(joint_chain, self.cga)
        root_to_target_distance = self.cga.point_distance(
            target_position, self.cga.to_point(self.cga.e_origin)
        )
        joint_chain_max_distance = joint_chain.max_distance()
        if root_to_target_distance > joint_chain_max_distance:
            # Unreachable target
            unreacheable_target_position = target_position
            target_position = self.cga.to_point(
                self.cga.normalize_vector(self.cga.to_vector(target_position))
                * joint_chain_max_distance
            )
            logging.warn(
                f"""Unreacheable target {self.cga.to_vector(unreacheable_target_position)}, the robot is able to extend 
                         at most {joint_chain_max_distance} meters but the 
                         target is at {root_to_target_distance} meters.
                         Setting the new target to {self.cga.to_vector(target_position)} as the closest to the 
                         desired target that is likely reacheable."""
            )
        # Likely reacheable target (within the joint chain sphere)
        end_effector_to_target_distance = self.cga.point_distance(
            target_position, point_chain[-1]
        )
        current_iteration = 1
        while (
            end_effector_to_target_distance > tolearance
            and current_iteration <= max_iterations
        ):
            # Fordward Reaching Stage
            point_chain[-1] = target_position
            for point_index in range(len(point_chain) - 1):
                self.fabrik_iteration(point_chain, joint_chain, point_index, True)
            # Backward Reaching Stage
            point_chain[0] = self.cga.to_point(self.cga.e_origin)
            for point_index in range(len(point_chain) - 1, 1, -1):
                self.fabrik_iteration(point_chain, joint_chain, point_index, False)
            end_effector_to_target_distance = self.cga.point_distance(
                target_position, point_chain[-1]
            )
        return point_chain

    def fabrik_iteration(self, point_chain, joint_chain, point_index, is_forward):
        iteration_type = "Forward" if is_forward else "Backward"
        logging.info(f"{iteration_type} point index {point_index}")
        target_index = point_index + 1 if is_forward else point_index - 1
        source_point = point_chain[point_index]
        target_point = point_chain[target_index]
        points_distance = self.cga.point_distance(source_point, target_point)
        joint_index = point_index if is_forward else point_index - 1
        joint_distance = joint_chain[joint_index].distance
        distance_ratio = joint_distance / points_distance
        # Linear Interpolation
        source_point = self.cga.to_point(
            ((1.0 - distance_ratio) * self.cga.to_vector(target_point))
            + (distance_ratio * self.cga.to_vector(source_point))
        )
        point_chain[point_index] = source_point

    def solve_old(self, joint_chain, target_position, max_iterations=100):
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

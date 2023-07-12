from fabrik.cga import ConformalGeometricAlgebra
from pytransform3d.transformations import (
    invert_transform,
)
import numpy as np
from dataclasses import dataclass
from fabrik.kinematics import UrdfRobotLibrary


@dataclass
class PoseSample:
    axis_index: int
    pose_origin_s: np.array
    pose_axis_s: np.array
    parameter_axis_index: int
    parameter_value: float


class FabrikSolver(object):
    def __init__(self):
        self.cga = ConformalGeometricAlgebra()
        self.resolution = 1e-10
        self.plane_bivector = self.cga.e1 ^ self.cga.e2

    def solve(
        self,
        open_chains,
        target_pose,
        tolearance=1e-6,
        max_iterations=100,
    ):
        """
        Notation:
            k: k-th joint
            n: number of joints
        """
        # move n-2 and take point sample
        """
        HERE I CAN DO A UNIT TEST TO VERIFY THIS 
        ACTUALLY WORKS
        """
        # connect point samples
        # get line from n-1 to zero pose
        # find dual point of interesection bewteen line and geometry
        """
        Check if any of the points of the dual point is within constraints
        if none is within constraints take the constrant closer to the line,
        otherwise take the point closer to zero pose
        """

    def pose_origin_in_space_to_reach_target_pose_given_parameters(
        self, open_chain, parameters, target_pose
    ):
        """
        Get the pose at origin seen in Space to reach the target pose,
        like moving the whole robot so that the end effector matches
        the target pose.
        Specifically, get the pose at origin, very potentially detached from the
        original pose zero, such that with the given parameters, it would
        have the same target pose.
        Pt=TokMkPo -> Po=inv(TokMk)Pt
        """
        return (
            invert_transform(open_chain.forward_transformation(parameters))
            @ target_pose
        )

    def solve_closest_points(self, positions, pose_target):
        points = [self.cga.point(*position.tolist()) for position in positions]
        circle = self.cga.circle_from_non_colinear_points(*points)
        vector_normal_to_circle = self.cga.normal_from_plane(
            self.cga.plane_from_non_colinear_points(*points)
        )
        pose_target = self.cga.point(*pose_target.tolist())
        vector_from_pose_target = self.cga.to_vector(pose_target)
        plane = self.cga.plane_from_two_vectors_and_point(
            vector_normal_to_circle, vector_from_pose_target, self.cga.e_origin
        )
        print("vector_normal_to_circle", vector_normal_to_circle)
        print("vector_from_pose_target", vector_from_pose_target)
        print("plane", self.cga.normal_from_plane(plane))
        print("plane", self.cga.normal_from_plane(plane) | self.cga.e_origin)
        print("plane", self.cga.normal_from_plane(plane) | points[0])
        # print("computing meet")
        dual_point = circle.meet(plane)
        center_position1, center_position2 = self.cga.project(dual_point)
        # NOTE: postions are w.r.t. the center of the circle as origin
        print(
            self.cga.to_vector(center_position1),
            "project",
            self.cga.to_vector(center_position2),
        )
        point1 = self.cga.sandwich(
            pose_target, self.cga.translator(self.cga.to_vector(center_position1))
        )
        point2 = self.cga.sandwich(
            pose_target, self.cga.translator(self.cga.to_vector(center_position2))
        )
        print(
            self.cga.to_vector(point1),
            "project",
            self.cga.to_vector(point2),
        )
        point1_distance_to_origin = point1 | self.cga.e_inf
        point2_distance_to_origin = point2 | self.cga.e_inf
        print(point1_distance_to_origin)
        print(point2_distance_to_origin)
        #if point1_distance_to_origin < point2_distance_to_origin:
        #    return point1_distance_to_origin
        #else:
        #    return point2_distance_to_origin
        #print(self.cga.to_vector(point1), "intersection", self.cga.to_vector(point2))
        #return point1

    def end_effector_to_target(
        self, open_chains, parameters, parameter_values, axis_index, target_pose
    ):
        chain = open_chains[axis_index]
        pose_samples = []
        for parameter_value in parameter_values:
            current_parameters = parameters.copy()
            current_parameters[axis_index] = parameter_value
            pose_origin_s = (
                self.pose_origin_in_space_to_reach_target_pose_given_parameters(
                    chain, current_parameters, target_pose
                )
            )
            """
            open_chains[axis_index]: pose of axis_index target-link.
                Affected by axis at "axis_index" and previous axis
            open_chains[axis_index - 1]: pose where the axis_index 
                is located. Affected by "axis_index - 1" and previous axis 
            open_chains[axis_index - 2]: pose where the axis previous 
                to axis_index is located. Affected by "axis_index - 2"
                and previous axis
            We are interested "axis_index - 2" as the 
                position at "axis_index" is fixed by the target position
                and the position at "axis_index - 1" is fixed by the
                target orientation. We want to know where "axis_index - 2"
                position is given changes in "axis_index" parameter
                guaranteeing target pose which locks in place 
                "axis_index" and "axis_index - 1"
            Note that the `current_parameters` might not be affected 
            by the `axis_index` as it will be further within the chain.
            The transformation is affected by the value in `axis_index`
            as the initial pose (`pose_origin_s`) is dependant on it. 
            """
            pre_pre_chain = open_chains[axis_index - 2]
            pose_pre_axis_s = (
                pre_pre_chain.forward_transformation(current_parameters) @ pose_origin_s
            )
            pose_sample = PoseSample(
                axis_index - 2,
                pose_origin_s,
                pose_pre_axis_s,
                axis_index,
                parameter_value,
            )
            pose_samples.append(pose_sample)
        return pose_samples


if __name__ == "__main__":
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.1)
    fabrik_solver = FabrikSolver()
    pose_target = np.array([200.0, 100.0, 0.0])
    poses = np.array(
        [
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, -20.0, 0.0],
        ]
    )
    solution = fabrik_solver.solve_closest_points(poses, pose_target)
    print("solution", solution)

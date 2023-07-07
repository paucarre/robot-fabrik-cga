from fabrik.cga import ConformalGeometricAlgebra
from fabrik.kinematics import UrdfRobotLibrary
from pytransform3d.transformations import (
    invert_transform,
)
from fabrik.kinematics import zero_pose
import numpy as np
from dataclasses import dataclass


@dataclass
class PoseSample:
    axis_index: int
    pose_origin_s: np.array
    pose_axis_s: np.array
    parameter_value: float


class FabrikRoboticsSolver(object):
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
            """
            pre_pre_chain = open_chains[axis_index - 2]
            pose_pre_axis_s = (
                pre_pre_chain.forward_transformation(current_parameters)
                @ pose_origin_s
            )
            pose_sample = PoseSample(
                axis_index, pose_origin_s, pose_pre_axis_s, parameter_value
            )
            pose_samples.append(pose_sample)
        return pose_samples


if __name__ == "__main__":
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.1)
    zeros = [0.0] * len(open_chains[-1])
    target_pose = open_chains[-1].forward_transformation(zeros) @ zero_pose()
    fabrik_solver = FabrikRoboticsSolver()
    low_limit = open_chains[-1].joint_limits[-1][0]
    high_limit = open_chains[-1].joint_limits[-1][1]
    mid_limit = low_limit + (0.9 * (high_limit - low_limit) / 2.0)
    values = [mid_limit]
    end_effector_axis_index = len(open_chains[-1]) - 1
    pose_samples = fabrik_solver.end_effector_to_target(
        open_chains, zeros, values, end_effector_axis_index, target_pose
    )
    """
    The test verifies the following:
    with the last parameter as the parameter_value
    and starting at pose pose_origin_s"
      - The end effector pose is the target pose
      - The pose at the axis before the last one ( 
       "the last axis moves the previous one"
      ) has pose pose_pre_pre_end_effector_s
    
    """
    """
    Verify that the end effector pose is the target pose
    """
    parameters = [0.0] * len(open_chains[-1])
    parameters[-1] = pose_samples[0].parameter_value
    end_effector_pose = (
        open_chains[-1].forward_transformation(parameters)
        @ pose_samples[0].pose_origin_s
    )
    print(end_effector_pose - target_pose)
    print(
        np.isclose(
            np.array(end_effector_pose), np.array(target_pose), rtol=1e-05, atol=1e-05
        ).all()
    )
    # print(pose_origin_s, pose_pre_pre_end_effector_s, parameter_value)
    """
    - The pose at the axis before the last one ( 
        "the last axis moves the previous one"
        ) has pose pose_pre_pre_end_effector_s
    """

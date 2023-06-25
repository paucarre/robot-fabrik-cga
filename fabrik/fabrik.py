from fabrik.cga import ConformalGeometricAlgebra
from fabrik.kinematics import UrdfRobotLibrary
from pytransform3d.transformations import (
    invert_transform,
    transform_from_exponential_coordinates,
)


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

    def end_effector_to_target(
        self,
        open_chains,
        target_pose,
        tolearance=1e-6,
        max_iterations=100,
    ):
        # get pose at origin given target pose wrt Space
        end_effector_chain = open_chains[-1]
        zeros = [0.0] * len(end_effector_chain)
        pose_origin_s = (
            invert_transform(end_effector_chain.forward_transformation(zeros))
            @ target_pose
        )
        # get pose at n-1 wrt Space, given new pose at origin
        pre_end_effector_chain = open_chains[-2]
        zeros = [0.0] * len(pre_end_effector_chain)
        pose_pre_end_effector_s = (
            pre_end_effector_chain.forward_transformation(zeros) @ pose_origin_s
        )
        # get the pose at n-2 wrt n-1, given new pose at origin
        pre_pre_end_effector_chain = open_chains[-3]
        zeros = [0.0] * len(pre_pre_end_effector_chain)
        pose_pre_pre_end_effector_s = (
            pre_pre_end_effector_chain.forward_transformation(zeros) @ pose_origin_s
        )
        pose_pre_pre_end_effector_pre_ef = (
            invert_transform(pre_end_effector_chain) @ pose_pre_pre_end_effector_s
        )
        # get screw in n-1 acting on n-2 pose
        low_limit = pre_end_effector_screw * pre_end_effector_chain.limits[-1][0]
        high_limit = pre_end_effector_screw * pre_end_effector_chain.limits[-1][1]
        mid_limit = low_limit + ((high_limit - low_limit) / 2.0)
        pre_end_effector_screw = pre_end_effector_chain[-1]
        transform_limit_low = (
            transform_from_exponential_coordinates(pre_end_effector_screw * low_limit)
            @ pose_pre_pre_end_effector_pre_ef
        )
        transform_limit_mid = (
            transform_from_exponential_coordinates(pre_end_effector_screw * mid_limit)
            @ pose_pre_pre_end_effector_pre_ef
        )
        transform_limit_high = (
            transform_from_exponential_coordinates(pre_end_effector_screw * high_limit)
            @ pose_pre_pre_end_effector_pre_ef
        )
        return transform_limit_mid, mid_limit


if __name__ == "__main__":
    fabrik_solver = FabrikRoboticsSolver()
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.1)

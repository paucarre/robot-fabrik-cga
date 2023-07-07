import unittest
import numpy as np
from fabrik.kinematics import UrdfRobotLibrary
import random
from fabrik.fabrik_solver import FabrikSolver
from fabrik.kinematics import zero_pose


class TestFabrikSolver(unittest.TestCase):
    def test_end_effector_to_target(self):
        urdf_robot = UrdfRobotLibrary.dobot_cr5()
        open_chains = urdf_robot.extract_open_chains(0.1)
        for _ in range(10):
            parameters = []
            for i in range(len(urdf_robot.joint_names)):
                parameters.append(
                    random.uniform(
                        urdf_robot.joint_limits[i][0], urdf_robot.joint_limits[i][1]
                    )
                )
            target_pose = (
                open_chains[-1].forward_transformation(parameters) @ zero_pose()
            )
            fabrik_solver = FabrikSolver()
            low_limit = open_chains[-1].joint_limits[-1][0]
            high_limit = open_chains[-1].joint_limits[-1][1]
            values = []
            for _ in range(10):
                parameter_value = random.uniform(low_limit, high_limit)
                values.append(parameter_value)
            end_effector_axis_index = len(open_chains[-1]) - 1
            pose_samples = fabrik_solver.end_effector_to_target(
                open_chains, parameters, values, end_effector_axis_index, target_pose
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
            for value, pose_sample in zip(values, pose_samples):
                parameter_axis_index = pose_sample.parameter_axis_index
                current_parameters = parameters.copy()
                current_parameters[parameter_axis_index] = pose_sample.parameter_value
                end_effector_pose = (
                    open_chains[parameter_axis_index].forward_transformation(
                        current_parameters
                    )
                    @ pose_sample.pose_origin_s
                )
                """
                check if reaching the target pose using the assigned parameter
                """
                assert np.isclose(
                    np.array(end_effector_pose),
                    np.array(target_pose),
                    rtol=1e-05,
                    atol=1e-05,
                ).all()
                assert end_effector_axis_index == pose_sample.parameter_axis_index
                assert value == pose_sample.parameter_value

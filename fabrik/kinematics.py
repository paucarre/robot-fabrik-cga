import numpy as np
from pytransform3d.transformations import (
    concat,
    transform_from_exponential_coordinates,
    exponential_coordinates_from_transform,
    invert_transform,
)
from pytransform3d.urdf import (
    UrdfTransformManager,
    parse_urdf,
    initialize_urdf_transform_manager,
)
from functools import reduce
import matplotlib.pyplot as plt


def zero_pose():
    return np.array([[1, 0, 0, 0], [0, 0, 0, 1]]).T


class OpenChainMechanism:
    def __init__(self, screws, initial_matrix, joint_limits):
        self.screws = screws
        self.initial_matrix = initial_matrix
        self.joint_limits = joint_limits

    def forward_transformation(self, coordinates):
        transformations = [
            transform_from_exponential_coordinates(screw * corordinate)
            for screw, corordinate in zip(self.screws, coordinates)
        ]
        transformations = reduce(
            lambda transform, computed_transform: concat(computed_transform, transform),
            transformations,
            np.eye(4),
        )
        return concat(self.initial_matrix, transformations)

    def __len__(self):
        return len(self.screws)

    def __getitem__(self, i):
        return self.screws[i]


class UrdfRobot:
    def __init__(self, name, links, joints):
        self.name = name
        self.links = links
        self.joints = joints
        self.link_names = [link.name for link in self.links]
        self.joint_names = [joint.joint_name for joint in self.joints]
        self.joint_axis = [joint.joint_axis for joint in self.joints]
        self.joint_types = [joint.joint_type for joint in self.joints]
        self.joint_transformation = [joint.child2parent for joint in self.joints]
        self.joint_limits = [joint.limits for joint in self.joints]
        self.tansform_manager = UrdfTransformManager()
        initialize_urdf_transform_manager(self.tansform_manager, name, links, joints)
        transform_indices = self.tansform_manager.to_dict()["transform_to_ij_index"]
        self.transform_indices = {
            joint_pair: index for joint_pair, index in transform_indices
        }

    def get_transform(self, i):
        link_source = self.link_names[i]
        link_destination = self.link_names[i + 1]
        transform_index = self.transform_indices[(link_destination, link_source)]
        transform = self.tansform_manager.to_dict()["transforms"][transform_index][1]
        transform = np.array(
            [transform[:4], transform[4:8], transform[8:12], transform[12:]],
            dtype=np.float32,
        )
        return transform

    def extract_open_chains(self, epsillon):
        open_chains = []
        screws = []
        transform_zero = np.eye(4)
        for i in range(len(self.link_names) - 1):
            for joint_name in self.joint_names:
                self.tansform_manager.set_joint(joint_name, 0.0)
            previous_transform_zero = transform_zero
            transform_zero = transform_zero @ self.get_transform(i)
            self.tansform_manager.set_joint(self.joint_names[i], epsillon)
            transform_epsillon = self.get_transform(i)
            """
                Mad = Tab(0)Tbc(0)Tcd(0)
                Tab(0)Tbc(0)Tcd(e) = Iexp(screw*e)Mad
                exp(screw*e) = Tab(0)Tbc(0)Tcd(e)inv(Mad)
            """
            exponential = (
                previous_transform_zero
                @ transform_epsillon
                @ invert_transform(transform_zero)
            )
            # coordinates = vee ( log ( exponential ) )
            coordinates = exponential_coordinates_from_transform(exponential)
            screw = coordinates / epsillon
            screws.append(screw)
            open_chain = OpenChainMechanism(
                screws.copy(), transform_zero, self.joint_limits[: i + 1]
            )
            open_chains.append(open_chain)
        return open_chains

    def transformations(self, values):
        for i, joint_name in enumerate(self.joint_names):
            self.tansform_manager.set_joint(joint_name, values[i])
        transform = np.eye(4)
        transformations = []
        for i in range(len(self.link_names) - 1):
            transform = transform @ self.get_transform(i)
            transformations.append(transform)
        return transformations

    def display(self):
        ax = self.tansform_manager.plot_frames_in(
            self.link_names[-1], s=0.1, whitelist=self.link_names, show_name=True
        )
        ax = self.tansform_manager.plot_connections_in(self.link_names[-1], ax=ax)
        self.tansform_manager.plot_visuals(self.link_names[-1], ax=ax)
        ax.set_xlim3d((-0.0, 0.25))
        ax.set_ylim3d((-0.1, 0.25))
        ax.set_zlim3d((0.1, 0.30))
        plt.show()

    def __repr__(self):
        return f"{self}"

    def __str__(self):
        return f"""Robot '{self.name}':
\t- Links: {" | ".join([f"{link.name}" for link in self.links])}
\t- Joints: {" | ".join([f"{joint.joint_name} {joint.joint_type}" for joint in self.joints])}"""


class UrdfRobotLibrary:
    def dobot_cr5():
        urdf_data = None
        with open("./urdf/cr5.urdf", "r") as urdf_file:
            urdf_data = urdf_file.read()
        name, links, joints = parse_urdf(
            urdf_data, mesh_path="./urdf/", package_dir="./urdf/", strict_check=True
        )
        return UrdfRobot(name, links, joints)


class Mechanisms:
    def robot_3r(l1, l2):
        screws = np.array(
            [[0, 0, 1, 0, 0, 0], [0, -1, 0, 0, 0, -l1], [1, 0, 0, 0, l2, 0]]
        )
        initial_matrix = np.array(
            [[0, 0, 1, l1], [0, 1, 0, 0], [-1, 0, 0, -l2], [0, 0, 0, 1]]
        )
        return OpenChainMechanism(screws, initial_matrix)


if __name__ == "__main__":
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.1)

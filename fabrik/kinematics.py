import numpy as np
from pytransform3d.transformations import concat, transform_from_exponential_coordinates
from pytransform3d.urdf import UrdfTransformManager, parse_urdf, initialize_urdf_transform_manager
from functools import reduce
import matplotlib.pyplot as plt

class OpenChainMechanism:
    def __init__(self, screws, initial_matrix):
        self.screws = screws
        self.initial_matrix = initial_matrix

    def forward_transformations(self, coordinates):
        transformations = [
            transform_from_exponential_coordinates(screw * corordinate)
            for screw, corordinate in zip(self.screws, coordinates)
        ]
        resulting_transformations = []
        for i in range(1, len(transformations)):
            current_transformations = transformations[:i]
            current_transformations = reduce(
                lambda transform, computed_transform: concat(
                    computed_transform, transform
                ),
                current_transformations,
                np.eye(4),
            )
            resulting_transformations.append(current_transformations)
        return resulting_transformations
        

    def forward_transformation(self, coordinates):
        transformations = [
            transform_from_exponential_coordinates(screw * corordinate)
            for screw, corordinate in zip(self.screws, coordinates)
        ]
        transformations = reduce(
            lambda transform, computed_transform: concat(
                computed_transform, transform
            ),
            transformations,
            np.eye(4),
        )
        return concat(
                self.initial_matrix, transformations
            )


class UrdfRobot():

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

    def display(self):
        ax = self.tansform_manager.plot_frames_in(
        self.link_names[-1], s=0.1, whitelist=self.link_names,
        show_name=True)
        ax = tansform_manager.plot_connections_in(self.link_names[-1], ax=ax)
        tansform_manager.plot_visuals(self.link_names[-1], ax=ax)
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
        with open('./urdf/cr5.urdf', 'r') as urdf_file:
            urdf_data = urdf_file.read()
        name, links, joints = parse_urdf(urdf_data, 
                                         mesh_path="./urdf/", 
                                         package_dir="./urdf/", 
                                         strict_check=True)
        return UrdfRobot(name, links, joints)
    
class Mechanisms:

    def robot_3r(l1, l2):
        screws = np.array([[0,  0, 1, 0,  0,  0],
                           [0, -1, 0, 0,  0,-l1],
                           [1,  0, 0, 0, l2,  0]])
        initial_matrix = np.array([[ 0,  0, 1,  l1],
                                   [ 0,  1, 0,   0],
                                   [-1,  0, 0, -l2],
                                   [ 0,  0, 0,   1]])
        return OpenChainMechanism(screws, initial_matrix)
    
if __name__ == "__main__":
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    tansform_manager = urdf_robot.tansform_manager
    urdf_robot.display()
    
    
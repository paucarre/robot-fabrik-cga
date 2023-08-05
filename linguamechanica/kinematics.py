import numpy as np
from pytransform3d.transformations import (
    exponential_coordinates_from_transform,
    invert_transform,
)
from pytransform3d.urdf import (
    UrdfTransformManager,
    parse_urdf,
    initialize_urdf_transform_manager,
)
import matplotlib.pyplot as plt
import torch
from pytorch3d import transforms


def to_left_multiplied(right_multiplied):
    """
     Converts matrix from right multiplied ( most common notation for SE3 )
     to left multiplied, which is the representation used in Pytorch 3D:
     M = [
        [Rxx, Ryx, Rzx, 0],
        [Rxy, Ryy, Rzy, 0],
        [Rxz, Ryz, Rzz, 0],
        [Tx,  Ty,  Tz,  1],
    ]
    This is equivalent to
     M = [
        [             ,  0],
        [ transpose(R),  0],
        [             ,  0],
        [      T      ,  1],
    ]
    """
    shape = right_multiplied.shape
    left_multiplied = right_multiplied.clone()
    if len(shape) == 3:
        left_multiplied = left_multiplied.transpose(1, 2)
        left_multiplied[:, 0:3, 0:3] = right_multiplied[:, 0:3, 0:3].transpose(1, 2)
    elif len(shape) == 2:
        left_multiplied = left_multiplied.transpose(0, 1)
        left_multiplied[0:3, 0:3] = right_multiplied[0:3, 0:3].transpose(0, 1)
    return left_multiplied


class DifferentiableOpenChainMechanism:
    def __init__(self, screws, initial_matrix, joint_limits):
        self.screws = screws
        self.initial_matrix = to_left_multiplied(initial_matrix)
        self.joint_limits = joint_limits

    def _jacobian_computation_forward(self, coords):
        transformation = self.forward_transformation(coords)
        twist = transforms.se3_log_map(transformation.get_matrix())
        return twist

    def compute_error_twist(self, coords, target_pose):
        current_transformation = self.forward_transformation(coords)
        target_transoformation = transforms.se3_exp_map(target_pose)
        current_trans_to_target = current_transformation.compose(
            transforms.Transform3d(matrix=target_transoformation).inverse()
        )
        error_twist = transforms.se3_log_map(current_trans_to_target.get_matrix())
        return error_twist

    def compute_weighted_error(error_twist, weights):
        return (error_twist * weights.unsqueeze(0)).sum(1)

    def inverse_kinematics(
        self,
        initial_coords,
        target_pose,
        min_error,
        error_weights,
        velocity_weights,
        max_steps=1000,
    ):
        current_coords = initial_coords
        error_twist = self.compute_error_twist(current_coords, target_pose)
        error = self.compute_weighted_error(error_twist, error_weights)
        velocity_rate = (
            torch.ones([target_pose.shape[0], 6, 1]).to(error.device) * velocity_weights
        )
        current_step = 0
        while error >= min_error and current_step < max_steps:
            jacobian = self.jacobian(current_coords)
            jacobian_pseudoinverse = torch.linalg.pinv(jacobian)
            parameter_delta = torch.bmm(jacobian_pseudoinverse, velocity_rate)
            current_coords += parameter_delta
            error_twist = self.compute_error_twist(current_coords, target_pose)
            error = DifferentiableOpenChainMechanism.compute_weighted_error(
                error_twist, error_weights
            )
            current_step += 1
        return current_coords

    def jacobian(self, coordinates):
        """
        From coordinates of shape:
            [ Batch, Coordinates ]
        Returns Jacobian of shape:
            [ Batch, Velocities, Coordinates]
        Velocities is always 6 with the
        first 3 components being translation
        and the last 3 rotation
        """
        jacobian = torch.autograd.functional.jacobian(
            self._jacobian_computation_forward, coordinates
        )
        """
        Dimensions:
            [batch, screw_coordinates, batch, coords]
        Need to be reduced to:
            [batch, screw_coordinates, coords]
        By using `take_along_dim`
        Conceptually this means that coordinates that are
        used in a kinematic chain are not used for other
        kinematic chains and thus the jacobian shall be zero.
        """
        selector = (
            torch.range(0, jacobian.shape[0] - 1)
            .long()
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .to(jacobian.device)
        )
        jacobian = torch.take_along_dim(jacobian, selector, dim=2).squeeze()
        return jacobian

    def forward_transformation(self, coordinates):
        twist = self.screws * coordinates.unsqueeze(2)
        original_shape = twist.shape
        twist = twist.view(-1, original_shape[2])
        transformations = transforms.se3_exp_map(twist)
        """
        Transformations will have indices of this type:
        [
            => The i-th index of the chain
            => The j-th chain ( it can be the same robot 
              with another pose or differnt robots 
              so long they have the same number of degres of freedom)
            => 4 rows of the left-transformation matrix
            => 4 columns of the left-transformation matrix
        ]
        """
        transformations = transformations.view(
            original_shape[0],
            original_shape[1],
            transformations.shape[1],
            transformations.shape[2],
        )
        chains_lenght = transformations.shape[1]
        num_chains = transformations.shape[0]
        computed_transforms = transforms.Transform3d(
            matrix=torch.eye(4).unsqueeze(0).repeat(num_chains, 1, 1)
        )
        for chain_idx in range(chains_lenght):
            current_transformations = transforms.Transform3d(
                matrix=transformations[:, chain_idx, :, :]
            )
            computed_transforms = current_transformations.compose(computed_transforms)
        initial_matrix = transforms.Transform3d(
            matrix=self.initial_matrix.unsqueeze(0).repeat(num_chains, 1, 1)
        )
        return initial_matrix.compose(computed_transforms)

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
        self.transform_manager = UrdfTransformManager()
        initialize_urdf_transform_manager(self.transform_manager, name, links, joints)
        transform_indices = self.transform_manager.to_dict()["transform_to_ij_index"]
        self.transform_indices = {
            joint_pair: index for joint_pair, index in transform_indices
        }

    def get_transform(self, i):
        link_source = self.link_names[i]
        link_destination = self.link_names[i + 1]
        transform_index = self.transform_indices[(link_destination, link_source)]
        transform = self.transform_manager.to_dict()["transforms"][transform_index][1]
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
                self.transform_manager.set_joint(joint_name, 0.0)
            previous_transform_zero = transform_zero
            transform_zero = transform_zero @ self.get_transform(i)
            self.transform_manager.set_joint(self.joint_names[i], epsillon)
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
            screws.append(
                np.expand_dims(np.concatenate([screw[3:], screw[:3]]), axis=0)
            )
            screw_torch = torch.Tensor(np.concatenate(screws.copy()))
            initial_torch = torch.Tensor(transform_zero)
            open_chain = DifferentiableOpenChainMechanism(
                screw_torch, initial_torch, self.joint_limits[: i + 1]
            )
            open_chains.append(open_chain)
        return open_chains

    def transformations(self, values):
        """
        This method assumes values is a batch of one element
        """
        assert values.shape[0] == 1
        for i, joint_name in enumerate(self.joint_names):
            self.transform_manager.set_joint(joint_name, values[0][i])
        transform = np.eye(4)
        transformations = []
        for i in range(len(self.link_names) - 1):
            transform = transform @ self.get_transform(i)
            transformations.append(transform)
        return transformations

    def display(self):
        ax = self.transform_manager.plot_frames_in(
            self.link_names[-1], s=0.1, whitelist=self.link_names, show_name=True
        )
        ax = self.transform_manager.plot_connections_in(self.link_names[-1], ax=ax)
        self.transform_manager.plot_visuals(self.link_names[-1], ax=ax)
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


if __name__ == "__main__":
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.1)

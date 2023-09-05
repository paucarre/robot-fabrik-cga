import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import transforms
from linguamechanica.environment import force_parameters_within_bounds


def get_pose_and_pose_error(state, open_chain):
    current_thetas = state[:, 6:]
    target_pose = state[:, :6]
    open_chain = open_chain.to(state.device)
    #print("get_pose_and_pose_error", state.shape, current_thetas.shape, target_pose.shape)
    error_pose = open_chain.compute_error_pose(current_thetas, target_pose)
    transformation = open_chain.forward_transformation(current_thetas)
    pose = transforms.se3_log_map(transformation.get_matrix())
    return pose, error_pose


def add_kinematics_to_state_embedding(state, open_chain):
    target_pose = state[:, :6]
    pose, pose_error = get_pose_and_pose_error(state, open_chain)
    return torch.cat([target_pose, pose, pose_error], 1)


def add_kinematics_to_state_action_embedding(state, action, open_chain):
    target_pose = state[:, :6]
    pose, pose_error = get_pose_and_pose_error(state, open_chain)
    new_state = state.detach().clone()
    new_state[:, 6:] += action
    force_parameters_within_bounds(new_state)
    pose_with_action, pose_error_with_action = get_pose_and_pose_error(
        new_state, open_chain
    )
    return torch.cat(
        [target_pose, pose, pose_error, pose_with_action, pose_error_with_action], 1
    )


class IKActor(nn.Module):
    def __init__(
        self,
        open_chain,
        max_action,
        initial_action_variance,
        max_variance,
        lr,
        action_dims,
        fc1_dims=1024,
        fc2_dims=512,
        fc3_dims=256,
        fc4_dims=256,
    ):
        super(IKActor, self).__init__()
        self.open_chain = open_chain
        # Adding 6 for pose and 6 for error pose wrt target
        input_dim = 6 * 3
        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        #self.mu = nn.Linear(fc4_dims, sum(action_dims) * 6)
        self.mu_other = nn.Linear(fc4_dims, sum(action_dims), bias=False)
        self.var = nn.Linear(fc4_dims, sum(action_dims))
        #self.scale = nn.Linear(fc2_dims, sum(action_dims))
        self.max_action = max_action
        self.initial_action_variance = initial_action_variance
        self.max_variance = max_variance
        #nn.init.normal_(self.fc1.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.fc1.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.fc2.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.fc2.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.fc3.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.fc3.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.fc4.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.fc4.bias.data, mean=0.0, std=1e-5)
        nn.init.normal_(self.mu_other.weight.data, mean=0.0, std=1e-1)
        nn.init.normal_(self.var.weight.data, mean=0.0, std=1e-10)
        self.initial_action_variance  = torch.Tensor([self.initial_action_variance])
        initial_variance_bias = torch.atanh((self.initial_action_variance *  2.0) - 1.0).item()
        nn.init.normal_(self.var.bias.data, mean=initial_variance_bias, std=1e-3)
        #self.var.bias.data = 0.00001
        #nn.init.normal_(self.mu.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.mu.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.scale.weight.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.scale.bias.data, mean=0.0, std=1e-5)
        # TODO: this should be more elegant
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        #print("state", state.shape)
        kinematic_embedding = add_kinematics_to_state_embedding(state, self.open_chain)
        x = F.relu(self.fc1(kinematic_embedding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # scale = F.relu(self.scale(x))
        mu = F.tanh(self.mu_other(x)) * self.max_action
        #mu = self.mu(x)
        #mu = mu.view([mu.shape[0], 6, -1])
        # TODO: remove the "0.0 * "
        var = ((F.tanh(self.var(x)) + 1.0) / 2.0) * self.max_variance
        #_, error_pose = get_pose_and_pose_error(state, self.open_chain)
        #mu = torch.bmm(mu, error_pose.unsqueeze(2))
        #mu = mu.squeeze(2)# * scale
        return mu, var


class PseudoinvJacobianIKActor(nn.Module):
    def __init__(
        self,
        open_chain,
    ):
        super(PseudoinvJacobianIKActor, self).__init__()
        self.open_chain = open_chain
        # TODO: do this in a more elegant way
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        We ignore the current pose from the
        state as we only care about the current parameters
        """
        current_thetas = state[:, 6:]
        target_pose = state[:, :6]
        self.open_chain = self.open_chain.to(state.device)
        error_pose = self.open_chain.compute_error_pose(current_thetas, target_pose)
        # TODO: the constant factor should be something else
        mu = -0.01 * self.open_chain.inverse_kinematics_step(current_thetas, error_pose)
        var = torch.zeros(mu.shape).to(mu.device)
        return mu, var


class Critic(nn.Module):
    def __init__(self, lr, action_dims, open_chain):
        super(Critic, self).__init__()
        input_dim = 6 * 5
        # Q1
        self.q1_l1 = nn.Linear(input_dim, 1024)
        self.q1_l2 = nn.Linear(1024, 512)
        self.q1_l3 = nn.Linear(512, 512)
        self.q1_l4 = nn.Linear(512, 256)
        self.q1_l5 = nn.Linear(256, 1, bias=False)
        nn.init.normal_(self.q1_l5.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q1_l4.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q1_l4.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q1_l3.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q1_l3.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q1_l2.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q1_l2.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q1_l1.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q1_l1.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q1_l5.bias.data, mean=0.0, std=1e-5)

        # Q2
        self.q2_l1 = nn.Linear(input_dim, 1024)
        self.q2_l2 = nn.Linear(1024, 512)
        self.q2_l3 = nn.Linear(512, 512)
        self.q2_l4 = nn.Linear(512, 256)
        self.q2_l5 = nn.Linear(256, 1, bias=False)
        nn.init.normal_(self.q2_l5.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q2_l5.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q2_l4.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q2_l4.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q2_l3.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q2_l3.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q2_l2.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q2_l2.weight.data, mean=0.0, std=1e-3)
        #nn.init.normal_(self.q2_l1.bias.data, mean=0.0, std=1e-5)
        #nn.init.normal_(self.q2_l1.weight.data, mean=0.0, std=1e-3)

        self.open_chain = open_chain

    def forward(self, state, action):
        kinematic_action_embedding = add_kinematics_to_state_action_embedding(
            state, action, self.open_chain
        )

        q1_x = F.relu(self.q1_l1(kinematic_action_embedding))
        q1_x = F.relu(self.q1_l2(q1_x))
        q1_x = F.relu(self.q1_l3(q1_x))
        q1_x = F.relu(self.q1_l4(q1_x))
        q1_x = -F.relu(self.q1_l5(q1_x))

        q2_x = F.relu(self.q2_l1(kinematic_action_embedding))
        q2_x = F.relu(self.q2_l2(q2_x))
        q2_x = F.relu(self.q2_l3(q2_x))
        q2_x = F.relu(self.q2_l4(q2_x))
        q2_x = -F.relu(self.q2_l5(q2_x))

        return q1_x, q2_x

    def Q1(self, state, action):
        kinematic_action_embedding = add_kinematics_to_state_action_embedding(
            state, action, self.open_chain
        )

        q1_x = F.relu(self.q1_l1(kinematic_action_embedding))
        q1_x = F.relu(self.q1_l2(q1_x))
        q1_x = F.relu(self.q1_l3(q1_x))
        q1_x = F.relu(self.q1_l4(q1_x))
        q1_x = -F.relu(self.q1_l5(q1_x))
        return q1_x

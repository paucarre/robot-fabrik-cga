import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch3d import transforms


class IKActor(nn.Module):
    def __init__(
        self,
        open_chain,
        max_action,
        min_variance,
        max_variance,
        lr,
        state_dims,
        action_dims,
        fc1_dims=256,
        fc2_dims=256,
    ):
        super(IKActor, self).__init__()
        self.open_chain = open_chain
        # Adding 6 for pose and 6 for error pose wrt target
        self.fc1 = nn.Linear(state_dims[0] + 6 + 6, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, sum(action_dims))
        self.var = nn.Linear(fc2_dims, sum(action_dims))
        self.max_action = max_action
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.jacobian_proportion = 1.0
        self.mu.bias.data.fill_(0.0)
        self.mu.weight.data.fill_(0.0)

    def forward(self, state):
        current_coords = state[:, 6:]
        target_pose = state[:, :6]
        self.open_chain = self.open_chain.to(state.device)
        error_pose = self.open_chain.compute_error_pose(current_coords, target_pose)
        transformation = self.open_chain.forward_transformation(current_coords)
        pose = transforms.se3_log_map(transformation.get_matrix())
        x = F.relu(self.fc1(torch.cat([state, pose, error_pose], 1)))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.mu(x)) * self.max_action
        var = (
            self.min_variance + ((F.tanh(self.var(x)) + 1.0) / 2.0) * self.max_variance
        )
        if self.jacobian_proportion is not None and self.jacobian_proportion > 0.0:
            error_pose = self.open_chain.compute_error_pose(current_coords, target_pose)
            jacobian_mu = -0.01 * self.open_chain.inverse_kinematics_step(
                current_coords, error_pose
            )
            var = torch.zeros(jacobian_mu.shape).to(jacobian_mu.device)
            mu = (
                (1.0 - self.jacobian_proportion) * mu
            ) + self.jacobian_proportion * jacobian_mu
            return jacobian_mu, var
        return mu, var


class DeterministicDiffIKActor(nn.Module):
    def __init__(
        self,
        open_chain,
        max_action,
    ):
        super(DeterministicDiffIKActor, self).__init__()
        self.open_chain = open_chain
        self.max_action = max_action
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        We ignore the current pose from the
        state as we only carre about the current parameters
        """
        target_pose = state[:6]
        current_coords = state[6:]
        error_pose = self.open_chain.compute_error_pose(current_coords, target_pose)
        mu = self.open_chain.inverse_kinematics_step(current_coords, error_pose)
        return mu, None


class Critic(nn.Module):
    def __init__(self, lr, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(sum(state_dim) + sum(action_dim), 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3.bias.data.fill_(0.0)
        self.l3.weight.data.fill_(0.0)

        # Q2
        self.l4 = nn.Linear(sum(state_dim) + sum(action_dim), 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.l6.bias.data.fill_(0.0)
        self.l6.weight.data.fill_(0.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = -F.relu(self.l3(q1))

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = -F.relu(self.l6(q2))
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = -F.relu(self.l3(q1))
        return q1

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, state_dims, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        """
            State dims should be for now:
                - Target pose, 6 
                - Current parameters, 6
                - Current parameter index, 1
            Action size should be:
                - Angle: sigmoid(x) - 0.5 or something similar
        """
        state_size = sum(state_dims)

        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 64)
        self.V = nn.Linear(64, 1)
        self.A = nn.Linear(64, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))

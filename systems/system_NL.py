import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

__all__ = ['num_dim_x', 'num_dim_control', 'f_func', 'DfDx_func', 'B_func', 'DBDx_func', 'Fa_func_np']

# torch.set_default_tensor_type('torch.DoubleTensor')

rho = 1.225
gravity = 9.81
drone_height = 0.09
mass = 1.47                  # mass

Sim_duration = 1000

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        if not x.is_cuda:
            self.cpu()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def read_weight(filename):
    model_weight = torch.load(filename, map_location=torch.device('cpu'))
    model = Network().double()
    model.load_state_dict(model_weight)
    model = model.float()
    # .cuda()
    return model

num_dim_x = 6
num_dim_control = 3

Fa_model = read_weight('data/Fa_net_12_3_full_Lip16.pth')

def Fa_func(z, vx, vy, vz):
    if next(Fa_model.parameters()).device != z.device:
        Fa_model.to(z.device)
    bs = z.shape[0]
    # use prediction from NN as ground truth
    state = torch.zeros([bs, 1, 12]).type(z.type())
    state[:,0,0] = z + drone_height
    state[:,0,1] = vx # velocity
    state[:,0,2] = vy # velocity
    state[:,0,3] = vz # velocity
    state[:,0,7] = 1.0
    state[:,0,8:12] = 6508.0/8000

    Fa = Fa_model(state).squeeze(1) * torch.tensor([30., 15., 10.]).reshape(1,3).type(z.type())
    return Fa

def Fa_func_np(x):
    z = torch.tensor(x[2]).float().view(1,-1)
    vx = torch.tensor(x[3]).float().view(1,-1)
    vy = torch.tensor(x[4]).float().view(1,-1)
    vz = torch.tensor(x[5]).float().view(1,-1)
    Fa = Fa_func(z, vx, vy, vz).cpu().detach().numpy()
    return Fa

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, y, z, vx, vy, vz = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = vx
    f[:, 1, 0] = vy
    f[:, 2, 0] = vz

    Fa = Fa_func(z, vx, vy, vz)
    f[:, 3, 0] = Fa[:, 0] / mass
    f[:, 4, 0] = Fa[:, 1] / mass
    f[:, 5, 0] = Fa[:, 2] / mass - gravity
    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')
#     # x: bs x n x 1
#     # f: bs x n x 1
#     # ret: bs x n x n
#     bs = x.shape[0]
#
#     x, y, theta, v = [x[:,i,0] for i in range(num_dim_x)]
#
#     J = torch.zeros(bs, num_dim_x, num_dim_x).type(x.type())
#
#     J[:, 0, 2] = - v * torch.sin(theta)
#     J[:, 1, 2] = v * torch.cos(theta)
#
#     J[:, 0, 3] = torch.cos(theta)
#     J[:, 1, 3] = torch.sin(theta)
#
#     return J

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 3, 0] = 1
    B[:, 4, 1] = 1
    B[:, 5, 2] = 1
    return B

def DBDx_func(x):
    # B: bs x n x m
    # ret: bs x n x n x m
    bs = x.shape[0]
    return torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())

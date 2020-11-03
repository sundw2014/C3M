import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

__all__ = ['num_dim_x', 'num_dim_control', 'f_func', 'DfDx_func', 'B_func', 'DBDx_func', 'Bbot_func']

# torch.set_default_tensor_type('torch.DoubleTensor')

mc = 1.
mp = 1.
g = 9.8
l = 1.

num_dim_x = 4
num_dim_control = 1

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v
    f[:, 1, 0] = omega
    f[:, 2, 0] = mp * torch.sin(theta) * (l * (omega ** 2) - g * torch.cos(theta)) / (mc + mp * (torch.sin(theta) ** 2))
    f[:, 3, 0] = (mp * l * (omega ** 2) * torch.cos(theta) * torch.sin(theta) - (mc + mp) * g * torch.sin(theta)) / l / (mc + mp * (torch.sin(theta)**2))
    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 2, 0] = 1 / (mc + mp * (torch.sin(theta) ** 2))
    B[:, 3, 0] = torch.cos(theta) / l / (mc + mp * (torch.sin(theta) ** 2))
    return B

def DBDx_func(x):
    # B: bs x n x m
    # ret: bs x n x n x m
    raise NotImplemented('NotImplemented')

def Bbot_func(x):
    # Bbot: bs x n x m
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]
    Bbot = torch.zeros(bs, num_dim_x, num_dim_x-num_dim_control).type(x.type())

    Bbot[:, 0, 0] = 1
    Bbot[:, 1, 1] = 1
    Bbot[:, 2, 2] = torch.cos(theta) / l / (mc + mp * (torch.sin(theta) ** 2))
    Bbot[:, 3, 2] = - 1 / (mc + mp * (torch.sin(theta) ** 2))
    return Bbot

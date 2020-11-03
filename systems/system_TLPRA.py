import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

__all__ = ['num_dim_x', 'num_dim_control', 'f_func', 'DfDx_func', 'B_func', 'DBDx_func']

# torch.set_default_tensor_type('torch.DoubleTensor')

m1 = 0.8
m2 = 2.3
a1 = 1
a2 = 1
g = 9.8
alpha = (m1+m2) * (a1 **2)
beta = m2 * (a2 ** 2)
eta = m2 * a1 * a2
e1 = g / a1

num_dim_x = 4
num_dim_control = 2

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    q1, q2, v1, v2 = [x[:,i,0] for i in range(num_dim_x)]

    Bbot_inv = torch.zeros(bs, num_dim_control, num_dim_control).type(x.type())
    Bbot_inv[:, 0, 0] = alpha + beta + 2 * eta * torch.cos(q2)
    Bbot_inv[:, 0, 1] = beta + eta * torch.cos(q2)
    Bbot_inv[:, 1, 0] = beta + eta * torch.cos(q2)
    Bbot_inv[:, 1, 1] = beta

    fbot = torch.zeros(bs, num_dim_control, 1).type(x.type())
    fbot[:, 0, 0] = -eta * (2 * v1 * v2 + v2 ** 2) * torch.sin(q2) + alpha * e1 * torch.cos(q1) + eta * e1 * torch.cos(q1+q2)
    fbot[:, 1, 0] = eta * (v1 ** 2) * torch.sin(q2) + eta * e1 * torch.cos(q1+q2)

    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v1
    f[:, 1, 0] = v2
    f[:, 2:, :] = torch.inverse(Bbot_inv).matmul(-fbot)
    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    q1, q2, v1, v2 = [x[:,i,0] for i in range(num_dim_x)]

    Bbot_inv = torch.zeros(bs, num_dim_control, num_dim_control).type(x.type())
    Bbot_inv[:, 0, 0] = alpha + beta + 2 * eta * torch.cos(q2)
    Bbot_inv[:, 0, 1] = beta + eta * torch.cos(q2)
    Bbot_inv[:, 1, 0] = beta + eta * torch.cos(q2)
    Bbot_inv[:, 1, 1] = beta

    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 2:, :] = torch.inverse(Bbot_inv)
    return B

def DBDx_func(x):
    # B: bs x n x m
    # ret: bs x n x n x m
    raise NotImplemented('NotImplemented')

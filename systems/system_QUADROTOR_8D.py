import torch

num_dim_x = 8
num_dim_control = 3

g = 9.81

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, y, z, vx, vy, vz, theta_x, theta_y = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = vx
    f[:, 1, 0] = vy
    f[:, 2, 0] = vz
    f[:, 3, 0] = g * torch.tan(theta_x)
    f[:, 4, 0] = g * torch.tan(theta_y)
    f[:, 5, 0] = 0
    f[:, 6, 0] = 0
    f[:, 7, 0] = 0

    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 5, 0] = 1
    B[:, 6, 1] = 1
    B[:, 7, 2] = 1
    return B

def DBDx_func(x):
    raise NotImplemented('NotImplemented')

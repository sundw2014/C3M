import torch

num_dim_x = 4
num_dim_control = 2

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, y, theta, v = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v * torch.cos(theta)
    f[:, 1, 0] = v * torch.sin(theta)
    f[:, 2, 0] = 0
    f[:, 3, 0] = 0
    return f

def DfDx_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    # ret: bs x n x n
    bs = x.shape[0]

    x, y, theta, v = [x[:,i,0] for i in range(num_dim_x)]

    J = torch.zeros(bs, num_dim_x, num_dim_x).type(x.type())

    J[:, 0, 2] = - v * torch.sin(theta)
    J[:, 1, 2] = v * torch.cos(theta)

    J[:, 0, 3] = torch.cos(theta)
    J[:, 1, 3] = torch.sin(theta)

    return J

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 2, 0] = 1
    B[:, 3, 1] = 1
    return B

def DBDx_func(x):
    # B: bs x n x m
    # ret: bs x n x n x m
    bs = x.shape[0]
    return torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())

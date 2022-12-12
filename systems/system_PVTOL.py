import torch

num_dim_x = 6
num_dim_control = 2

m = 0.486;
J = 0.00383;
g = 9.81;
l = 0.25;

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    p_x, p_z, phi, v_x, v_z, dot_phi = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v_x * torch.cos(phi) - v_z * torch.sin(phi)
    f[:, 1, 0] = v_x * torch.sin(phi) + v_z * torch.cos(phi)
    f[:, 2, 0] = dot_phi
    f[:, 3, 0] = v_z * dot_phi - g * torch.sin(phi)
    f[:, 4, 0] = - v_x * dot_phi - g * torch.cos(phi)
    f[:, 5, 0] = 0
    return f

def DfDx_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    # ret: bs x n x n
    bs = x.shape[0]

    p_x, p_z, phi, v_x, v_z, dot_phi = [x[:,i,0] for i in range(num_dim_x)]
    J = torch.zeros(bs, num_dim_x, num_dim_x).type(x.type())
    J[:, 0, 2] = - v_x * torch.sin(phi) - v_z * torch.cos(phi)
    J[:, 1, 2] = v_x * torch.cos(phi) - v_z * torch.sin(phi)
    J[:, 2, 2] = 0
    J[:, 3, 2] = - g * torch.cos(phi)
    J[:, 4, 2] = g * torch.sin(phi)
    J[:, 5, 2] = 0
    J[:, 0, 3] = torch.cos(phi)
    J[:, 1, 3] = torch.sin(phi)
    J[:, 2, 3] = 0
    J[:, 3, 3] = 0
    J[:, 4, 3] = - dot_phi
    J[:, 5, 3] = 0
    J[:, 0, 4] = - torch.sin(phi)
    J[:, 1, 4] = torch.cos(phi)
    J[:, 2, 4] = 0
    J[:, 3, 4] = dot_phi
    J[:, 4, 4] = 0
    J[:, 5, 4] = 0
    J[:, 0, 5] = 0
    J[:, 1, 5] = 0
    J[:, 2, 5] = 1
    J[:, 3, 5] = v_z
    J[:, 4, 5] = - v_x
    J[:, 5, 5] = 0

    return J

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 4, 0] = 1 / m
    B[:, 4, 1] = 1 / m
    B[:, 5, 0] = l / J
    B[:, 5, 1] = -l / J
    return B

def DBDx_func(x):
    # B: bs x n x m
    # ret: bs x n x n x m
    bs = x.shape[0]
    return torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())

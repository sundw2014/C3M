import torch

num_dim_x = 9
num_dim_control = 3

g = 9.81;

# the following dynamics is converted from the original Matlab code
# we noticed that xc(10) is a dead state variable
# b_T =  [sin(xc(9)); -cos(xc(9))*sin(xc(8)); cos(xc(9))*cos(xc(8))];
#
# f_ctrl =     [xc(4:6);
#               [0;0;g] - xc(7)*b_T;
#               zeros(4,1)];
# B_ctrl = @(xc)[zeros(6,4);
#                eye(4)];

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = x4
    f[:, 1, 0] = x5
    f[:, 2, 0] = x6

    f[:, 3, 0] = - x7 * torch.sin(x9)
    f[:, 4, 0] = x7 * torch.cos(x9) * torch.sin(x8)
    f[:, 5, 0] = g - x7 * torch.cos(x9) * torch.cos(x8)

    f[:, 6, 0] = 0
    f[:, 7, 0] = 0
    f[:, 8, 0] = 0

    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 6, 0] = 1
    B[:, 7, 1] = 1
    B[:, 8, 2] = 1
    return B

def DBDx_func(x):
    raise NotImplemented('NotImplemented')

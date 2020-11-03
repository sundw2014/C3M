import torch
from torch import nn
from torch.autograd import grad
import numpy as np
from tianshou.data import to_torch

effective_dim_start = 2
effective_dim_end = 4

class ActorProb(nn.Module):
    def __init__(self, n, m, device='cpu'):
        super().__init__()
        self.device = device
        self.n = n
        self.m = m
        c = 3 * n
        self.model_w1 = torch.nn.Sequential(
            torch.nn.Linear(2*(effective_dim_end - effective_dim_start), 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, c*n, bias=True))

        self.model_w2 = torch.nn.Sequential(
            torch.nn.Linear(2*(effective_dim_end - effective_dim_start), 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, m*c, bias=True))

        self.sigma = nn.Parameter(torch.zeros(m, 1))

        self.to(device)

    def forward(self, s, state=None, info={}):
        self.device = next(self.parameters()).device
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.flatten(1)

        bs = s.shape[0]
        w1 = self.model_w1(torch.cat([s[:, effective_dim_start:effective_dim_end], s[:, self.n+effective_dim_start:self.n+effective_dim_end]], dim=1)).reshape(bs, -1, self.n)
        w2 = self.model_w2(torch.cat([s[:, effective_dim_start:effective_dim_end], s[:, self.n+effective_dim_start:self.n+effective_dim_end]], dim=1)).reshape(bs, self.m, -1)
        mu = w2.matmul(torch.tanh(w1.matmul((s[:,:self.n] - s[:,self.n:]).unsqueeze(-1)))).squeeze(-1)

        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()

        return (mu, sigma), None

# closure for RL
class CONTROLLER_FUNC(nn.Module):
    """docstring for CONTROLLER_FUNC."""

    def __init__(self, actor):
        super(CONTROLLER_FUNC, self).__init__()
        self.actor = actor

    def forward(self, x, xe, uref):
        (mu, _), _ = self.actor(torch.cat([x, x-xe], dim=1).squeeze(-1))
        mu = mu.unsqueeze(-1) + uref
        return mu

# closure for ours
class U_FUNC(nn.Module):
    """docstring for U_FUNC."""

    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        w1 = self.model_u_w1(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, -1, self.num_dim_x)
        w2 = self.model_u_w2(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref

        return u

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda = False):
    model_Wbot = torch.nn.Sequential(
        torch.nn.Linear(1, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, (num_dim_x-num_dim_control) ** 2, bias=False))

    dim = effective_dim_end - effective_dim_start
    model_W = torch.nn.Sequential(
        torch.nn.Linear(dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_x * num_dim_x, bias=False))

    c = 3 * num_dim_x
    model_u_w1 = torch.nn.Sequential(
        torch.nn.Linear(2*dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c*num_dim_x, bias=True))

    model_u_w2 = torch.nn.Sequential(
        torch.nn.Linear(2*dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_control*c, bias=True))

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        W = model_W(x[:,effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
        Wbot = model_Wbot(torch.ones(bs, 1).type(x.type())).view(bs, num_dim_x-num_dim_control, num_dim_x-num_dim_control)
        W[:, 0:num_dim_x-num_dim_control, 0:num_dim_x-num_dim_control] = Wbot
        W[:, num_dim_x-num_dim_control::, 0:num_dim_x-num_dim_control] = 0

        # W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)

        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
        return W


    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func

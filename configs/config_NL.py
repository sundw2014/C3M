import numpy as np
from utils import temp_seed

# for ours training
X_MIN = np.array([-5., -5., 0., -1., -1., -1.]).reshape(-1,1)
X_MAX = np.array([5., 5., 2., 1., 1., 1.]).reshape(-1,1)

lim = 1.
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1,1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1,1)

UREF_MIN = np.array([-1.,-1.,-3.]).reshape(-1,1)
UREF_MAX = np.array([ 1., 1., 9.]).reshape(-1,1)

# for sampling ref
X_INIT_MIN = np.array([-3., -3., 0.5, 1., 0., 0.])
X_INIT_MAX = np.array([ 3.,  3.,  1., 1., 0., 0.])

XE_INIT_MIN = np.array([-1, -1, -0.4, -1., -1., 0.])
XE_INIT_MAX = np.array([ 1,  1.,  1.,  1.,  1., 0.])

time_bound = 3.
time_step = 0.03
t = np.arange(0, time_bound, time_step)

state_weights = np.array([1,1,1,0.1,0.1,0.1])

gravity = 9.81
mass = 1.47                  # mass

import system_NL as system

def system_reset(seed):
    SEED_MAX = 10000000
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        x_0 = xref_0 + xe_0
        Fa = system.Fa_func_np(xref_0.reshape(-1)).reshape(-1)

        freqs = list(range(1,10+1))
        # freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (0.5 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        uref = []
        for _t in t:
            u = np.array([0,0,gravity]) - Fa/mass # ref
            for freq, weight in zip(freqs, weights):
                u += np.array([weight[0] * np.sin(freq * _t/time_bound * 2*np.pi), weight[1] * np.sin(freq * _t/time_bound * 2*np.pi), weight[2] * np.sin(freq * _t/time_bound * 2*np.pi)])
            # u += 0.01*np.random.randn(2)
            uref.append(u)

    return x_0, xref_0, uref

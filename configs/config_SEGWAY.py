import numpy as np
from utils import temp_seed

# for ours training
X_MIN = np.array([-5., -np.pi/3, -1., -np.pi]).reshape(-1,1)
X_MAX = np.array([ 5.,  np.pi/3,  1.,  np.pi]).reshape(-1,1)

UREF_MIN = np.array([0.,]).reshape(-1,1)
UREF_MAX = np.array([0.,]).reshape(-1,1)

lim = 1.
XE_MIN = np.array([-lim, -lim, -lim, -lim]).reshape(-1,1)
XE_MAX = np.array([lim, lim, lim, lim]).reshape(-1,1)

# for sampling ref
# X_INIT_MIN = np.array([-4., -np.pi/4, -0.5, -np.pi/2])
# X_INIT_MAX = np.array([ 4.,  np.pi/4,  0.5,  np.pi/2])

X_INIT_MIN = np.array([0., 0, 0., 0])
X_INIT_MAX = np.array([0., 0, 0., 0])

XE_INIT_MIN = np.array([-1., -np.pi/3, -0.5, -np.pi])
XE_INIT_MAX = np.array([ 1.,  np.pi/3,  0.5,  np.pi])

time_bound = 6.
time_step = 0.03
t = np.arange(0, time_bound, time_step)

state_weights = np.array([1,1,0.1,0.1])

def system_reset(seed):
    SEED_MAX = 10000000
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        x_0 = xref_0 + xe_0

        # freqs = list(range(2,10+1))
        freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (0. * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        uref = []
        for _t in t:
            u = np.array([10.2 * xref_0[2] / 47.9]) # ref
            for freq, weight in zip(freqs, weights):
                u += np.array([weight[0] * (-1) ** (int(freq * _t/time_bound)) * np.sin(freq * _t/time_bound * 2*np.pi),])
            # u += 0.01*np.random.randn(2)
            uref.append(u)
        # from IPython import embed; embed()
    return x_0, xref_0, uref

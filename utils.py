import gzip
import pickle
import contextlib
import numpy as np

def savepklz(data_to_dump, dump_file_full_name):
    ''' Saves a pickle object and gzip it '''

    with gzip.open(dump_file_full_name, 'wb') as out_file:
        pickle.dump(data_to_dump, out_file)


def loadpklz(dump_file_full_name):
    ''' Loads a gziped pickle object '''

    with gzip.open(dump_file_full_name, 'rb') as in_file:
        dump_data = pickle.load(in_file)

    return dump_data

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def EulerIntegrate(controller, f, B, xstar, ustar, xinit, t_max = 10, dt = 0.05, with_tracking = False, sigma = 0., noise_bound = None):
    t = np.arange(0, t_max, dt)

    trace = []
    u = []

    xcurr = xinit
    trace.append(xcurr)

    for i in range(len(t)):
        if with_tracking:
            xe = xcurr - xstar[i]
        ui = controller(xcurr, xe, ustar[i]) if with_tracking else ustar[i]
        if with_tracking:
            # print(xcurr.reshape(-1), xstar[i].reshape(-1), ui.reshape(-1))
            pass

        if not noise_bound:
            noise_bound = 3 * sigma
        noise = np.random.randn(*xcurr.shape) * sigma
        noise[noise>noise_bound] = noise_bound
        noise[noise<-noise_bound] = -noise_bound

        dx = f(xcurr) + B(xcurr).dot(ui) + noise
        xnext =  xcurr + dx*dt
        # xnext[xnext>100] = 100
        # xnext[xnext<-100] = -100

        trace.append(xnext)
        u.append(ui)
        xcurr = xnext
    return trace, u

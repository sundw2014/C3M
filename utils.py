import numpy as np

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

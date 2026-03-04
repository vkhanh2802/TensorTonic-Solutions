import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w_t = np.array(w)
    g_t = np.array(g)
    s_t = np.array(s)
    s_new = beta*s_t + (1-beta)*(g_t**2)
    w_new = w_t - (lr*g_t)/(np.sqrt(s_new)+eps)
    return w_new, s_new
import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    w_t = np.array(w)
    g_t = np.array(g)
    G_t = np.array(G)
    G_new = G_t + g_t**2
    hed = np.sqrt(G_new+eps)
    w_new = w_t - (lr * g_t)/ hed
    return w_new, G_new
import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param_1 = np.array(param)
    grad_1 = np.array(grad)
    m_1 = np.array(m)
    v_1 = np.array(v)
    
    m_t = beta1*m_1 + (1-beta1)*grad_1
    v_t = beta2*v_1 + (1-beta2)*(grad_1**2)
    m_new = m_t / (1-beta1**t)
    v_new = v_t /(1-beta2**t)
    param_new = param_1 - lr*(m_new /(np.sqrt(v_new) + eps))
    return param_new, m_t, v_t
    
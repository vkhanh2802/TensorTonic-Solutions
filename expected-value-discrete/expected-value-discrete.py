import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    total=sum(p)
    if total!=1:
        raise ValueError("Khong hop le")
    x_new=np.array(x)
    p_new=np.array(p)
    
    return np.dot(x_new,p_new)
    

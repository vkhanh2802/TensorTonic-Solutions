
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x=0
    for i in range(steps):
        derivate=2*a*x0+b
        x=x0-lr*derivate
        x0=x
    return x0   
    
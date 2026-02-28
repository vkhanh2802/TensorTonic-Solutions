import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A_new=np.array(A)
    row, col =A_new.shape
    total=0
    for i in range(row):
        for j in range(col):
            if i==j:
                total+=A_new[i][j]           
    return total

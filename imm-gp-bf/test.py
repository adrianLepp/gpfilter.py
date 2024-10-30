import numpy as np


m = np.array([[-1.14283666e-05, -3.17565931e-06, -5.53818213e-06],
 [-3.26486561e-05,  3.20411063e-06, -5.53637926e-06],
 [-3.26358877e-05, -3.16108810e-06,  7.54912771e-06]])




def fix_nonpositive_definite(a):
    EPS = 1e-6;
    ZERO = 1e-10;
    [eigenval,eigenvec] = np.linalg.eig(a)
    
    for n in range(len(eigenval)):
        if eigenval[n] <= ZERO:
            eigenval[n] = EPS

    sigma = eigenvec * np.diag(eigenval) * eigenvec.T

    return sigma


print(fix_nonpositive_definite(m[0:1,0:1]))
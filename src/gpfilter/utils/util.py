import torch
import numpy as np
from scipy.linalg import cholesky


# %% mean std

def normalize_mean_std_np(data:np.ndarray, mean:float=None, std:float=None):
    if (mean is None or std is None):
        mean = data.mean(axis=0)
        std = data.std(axis=0)

    return (data - mean) / std, (mean, std)


def denormalize_mean_std(data:np.ndarray, mean: float, std:float):
    return (data - mean) / std

# %% numpoy min max
def normalize_min_max_np(data:np.ndarray, min_val:float=None, delta:float=None):
    
    if (min_val is None or delta is None):
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        delta = max_val - min_val
        
        # check for every dimension if diff=0 and replace with 1
        try:
            for i in range(delta.shape[0]):
                if delta[i] == 0:
                    delta[i] = 1
        except:
            if delta == 0:
                delta = 1

    normalized_data = (data - min_val) / delta #(max_val - min_val)
    return normalized_data, (min_val, delta)

def denormalize_min_max(data, min_val:float, delta:float):
    return data * delta + min_val

# %% pytorch min max

def normalize_min_max_torch(data:torch.tensor, min_val:float=None, delta:float=None):
    
    if (min_val is None or delta is None):
        min_val = data.min(dim=0)[0]
        max_val = data.max(dim=0)[0]
        delta = max_val - min_val
        
        # check for every dimension if diff=0 and replace with 1
        try:
            for i in range(delta.shape[0]):
                if delta[i] == 0:
                    delta[i] = 1
        except:
            if delta == 0:
                delta = 1

    normalized_data = (data - min_val) / delta
    return normalized_data, (min_val, delta)

# %% cholesky decomposition fix

def fix_nonpositive_definite(a):
    EPS = 1e-6;
    ZERO = 1e-10;
    [eigenval,eigenvec] = np.linalg.eig(a)
    
    for n in range(len(eigenval)):
        if eigenval[n] <= ZERO:
            eigenval[n] = EPS

    sigma = eigenvec * np.diag(eigenval) * eigenvec.T

    return sigma

def cholesky_fix(a, lower:bool=False, overwrite_a:bool=False, check_finite:bool=True):
    try:
        return cholesky(a, lower, overwrite_a, check_finite)
    except np.linalg.LinAlgError as inst:
        idx = int(inst.args[0][0])
        a[0:idx,0:idx] = fix_nonpositive_definite(a[0:idx,0:idx])
        return cholesky_fix(a, lower, overwrite_a, check_finite)
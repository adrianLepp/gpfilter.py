
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np
from GP_BF import GP_UKF

#--------------------------------------------------------------------------------
# init Filters
#------------------------------------------------------------------------------

def init_GP_UKF(x, fx, hx, n , Qfct, P, z_std, dt, alpha=.1, beta=2., kappa=-1):
    dim_x = x.shape[0]
    dim_z = x.shape[0]
    R = np.diag([z_std**2] * n)

    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(n, alpha, beta, kappa)

    gp_ukf = GP_UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, Qfct=Qfct, points=points)
    #              sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
    #              residual_x=None,
    #              residual_z=None,
    #              state_add=None)

    gp_ukf.x = x
    gp_ukf.P *= P
    gp_ukf.R = R 
    gp_ukf.Q = Qfct(x)

    return gp_ukf

def init_UKF(x, fx, hx , n, x_std, P, z_std, dt, alpha=.1, beta=2., kappa=-1):
    dim_x = x.shape[0]
    dim_z = x.shape[0]
    R = np.diag([z_std**2] * n)
    points = MerweScaledSigmaPoints(n, alpha, beta, kappa)

    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, points=points)

    ukf.x = x
    ukf.P *= P 
    ukf.R = R 
    ukf.Q = Q_discrete_white_noise(dim=n, dt=dt, var=x_std**2, block_size=1)

    return ukf
    

from filterpy.kalman import UnscentedKalmanFilter
import numpy as np
import GPy

class GP_SSM():
    def __init__(self, dx, y, n, l=1, sigma=1, sigmaY=1):
        kernel_list  = []
        X_list = []
        Y_list = []
        likelihood_list = []

        for i in range(n):
            se = GPy.kern.RBF(input_dim=n, lengthscale=l, variance=sigma)
            gauss = GPy.likelihoods.Gaussian(variance=sigmaY ** 2)

            kernel_list.append(se)
            likelihood_list.append(gauss)
            X_list.append(np.transpose(y))
            Y_list.append(np.transpose(np.expand_dims(dx[i,:], axis=0)))
        
        self.kernel = kernel_list
        self.likelihood = likelihood_list
        self.X = X_list
        self.Y = Y_list
        self.l = l
        self.sigma = sigma
        self.sigmaY = sigmaY
        self.n = n

        self.gp_ssm = GPy.models.MultioutputGP(
            X_list=X_list,
            Y_list=Y_list,
            kernel_list=kernel_list,
            likelihood_list=likelihood_list,
        )

    def optimize(self, msg=0, ipynb=False):
        self.gp_ssm.optimize(messages=msg, ipython_notebook=ipynb)

    #TODO predict returns mean and variance but we only need one at a time. 

    def stateTransition(self, xIn, dt):
        x = np.array([xIn])
        mu, var = self.gp_ssm.predict_noiseless(Xnew=[x]*self.n)
        
        return np.add(x, np.multiply(np.transpose(mu), dt))
    
    def stateTransitionVariance(self, xIn):
        x = np.array([xIn])
        mu, var = self.gp_ssm.predict_noiseless(Xnew=[x]*self.n)

        Q = np.diagflat(var)
        #TODO: is this correct or what should I do with dt?
        return Q

class GP_UKF(UnscentedKalmanFilter):
    #TODO: **args?
    def __init__(self, dim_x, dim_z, dt, hx, fx, points, Qfct,
        # sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
        # residual_x=None,
        # residual_z=None,
        # state_add=None
        ):

        super().__init__(dim_x, dim_z, dt, hx, fx, points,
            # sqrt_fn, x_mean_fn, z_mean_fn,
            # residual_x,
            # residual_z,
            # state_add
            )
        self.Qfct = Qfct
    
    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        self.Q = self.Qfct(self.sigmas_f[0,:])

        super().predict(dt, UT, fx)

def init_GP_UKF(x, fx, hx , points, Qfct, P, R, Q, dt):
    dim_x = x.shape[0]
    dim_z = R.shape[0]

    gp_ukf = GP_UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, Qfct=Qfct, points=points)
    #              sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
    #              residual_x=None,
    #              residual_z=None,
    #              state_add=None)

    gp_ukf.x = x
    gp_ukf.P *= P 
    gp_ukf.R = R 
    gp_ukf.Q = Q

    return gp_ukf
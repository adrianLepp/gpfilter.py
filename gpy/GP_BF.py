
from filterpy.kalman import UnscentedKalmanFilter
import numpy as np
import GPy

#TODO make a parent class for GP_SSM and GP_SSM2 so that i am forced to implement the same methods (like normalize and denormalize)
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
    

class GP_SSM2():
    '''
    This is a wrapper for a state space system for the GPMultioutRegression class in GPy which is an implementation of Latent Variable Multiple Output Gaussian Processes (LVMOGP) in [Dai_et_al_2017]
    Dai, Z.; Alvarez, M.A.; Lawrence, N.D: Efficient Modeling of Latent Information in Supervised Learning using Gaussian Processes. In NIPS, 2017.

    https://gpy.readthedocs.io/en/deploy/GPy.models.html#module-GPy.models.gp_multiout_regression

    Be careful with the notations:
    - I still use the notation where D is the number of samples and n is the state and measurement dimension
    - In the paper however, they use D for the number of outputs (different conditions) and N for the number of samples

    param dx: shape(D, n) where D is the number of samples and n is the output dimension
    param y: shape(D, n) where D is the number of samples and n is the output dimension
    '''
    def __init__(self, dx, y, n, l=1, sigma=1, sigmaY=1):

        self.l = l
        self.sigma = sigma
        self.sigmaY = sigmaY
        self.n = n

        D = n #   should be equal to y.shape[1] #output dimension
        Mr = D
        Mc = y.shape[0] #number of samples

        # WATCH OUT: latent dimension can have a tremendous impact on the performance (Qr=5 with threeTank guessed same Value for all states)
        Qr = 3
        Qc = y.shape[1] # input dimension

        #normalize
        self.dxMean = dx.mean(axis=0)
        self.dxStd = dx.std(axis=0)

        self.yMean = y.mean(axis=0)
        self.yStd = y.std(axis=0)

        dxNorm = (dx - self.dxMean) / self.dxStd
        yNorm = (y - self.yMean) / self.yStd

        self.gp_ssm = GPy.models.GPMultioutRegression(
            yNorm,
            dxNorm,
            Xr_dim=Qr, 
            kernel_row=GPy.kern.RBF(Qr,ARD=True), #TODO what is ARD doing 
            num_inducing=(Mc,Mr),
            init='GP'
        )        

    def optimize(self, msg=0, ipynb=False):
        self.gp_ssm.optimize_auto()

    #TODO predict returns mean and variance but we only need one at a time. 

    def stateTransition(self, xIn, dt):
        x = np.array([xIn]) # a 2D array is expected so we need to wrap it in another array
        xNorm = (x - self.yMean) / self.yStd
        mu, var = self.gp_ssm.predict_noiseless(Xnew=xNorm) #:type Xnew : np.ndarray (1,stateN)
        dx = mu * self.dxStd + self.dxMean
        return np.add(x, np.multiply(dx, dt))
    
    def stateTransitionVariance(self, xIn):
        xNorm = (np.array([xIn])-self.yMean) / self.yStd 
        mu, var = self.gp_ssm.predict_noiseless(Xnew=xNorm)

        Q = np.diagflat(var* self.dxStd + self.dxMean)
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
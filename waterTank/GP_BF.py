
from filterpy.kalman import UnscentedKalmanFilter
import numpy as np
import GPy
import torch
import gpytorch
from abc import ABC, abstractmethod
from util import normalize_min_max_np, normalize_min_max_torch, denormalize_min_max

DEBUG = False

class GP_SSM(ABC):
    
    def __init__(self, dxData, yData, n, normalize):
        self.n = n
        self.norm_param_x = None
        self.norm_param_y = None
        self.normFct  = None
        self.denormFct = None
            

        if normalize:
            self.normFct = normalize_min_max_np
            self.denormFct = denormalize_min_max
            
            self.x_train, self.norm_param_x =  self.normFct(yData)
            self.y_train, self.norm_param_y =  self.normFct(dxData)
        else:
            self.x_train = yData
            self.y_train = dxData

    
    @abstractmethod
    def optimize(self):
        pass
    @abstractmethod
    def stateTransition(self, xIn, dt):
        pass
        
    @abstractmethod
    def stateTransitionVariance(self, xIn):
        pass
    
    def normalize(self, x, param):
        if self.normFct:
            x, _ = self.normFct(x, *param)
        return x
    
    def denormalize(self, x, param):
        if self.denormFct:
            x = self.denormFct(x, *param)
        return x
        
'''
Simple Multioutput GP from the library gpy


TODO: adjust parameter for different kernel and likelihood
TODO: normalization

'''
class GP_SSM_gpy_multiout(GP_SSM):
    def __init__(self, dxData:np.ndarray, yData:np.ndarray, n:int, normalize=False, 
                 kern=GPy.kern.RBF, 
                 likelihood=GPy.likelihoods.Gaussian,
                 l=1, 
                 sigma=1, 
                 sigmaY=1
    ):
        super().__init__(dxData, yData, n, normalize)
        
        kernel_list  = []
        likelihood_list = []
        X_list = []
        Y_list = []

        for i in range(n):
            kernel_list.append(kern(input_dim=n, lengthscale=l, variance=sigma))
            likelihood_list.append(likelihood(variance=sigmaY ** 2))
            
            X_list.append(np.transpose(self.x_train))
            Y_list.append(np.transpose(np.expand_dims(self.y_train[i,:], axis=0)))
        
        
        self.gp = GPy.models.MultioutputGP(
            X_list=X_list,
            Y_list=Y_list,
            kernel_list=kernel_list,
            likelihood_list=likelihood_list,
        )

    def optimize(self, iterations=50, verbose=False):
        self.gp.optimize(messages=verbose, ipython_notebook=False, max_iters=iterations)

    def stateTransition(self, xIn, dt):
        xIn = np.array([xIn])
        x = self.normalize(xIn, self.norm_param_x)
            
        dx, _ = self.gp.predict_noiseless(Xnew=[x]*self.n)
        
        dx = self.denormalize(dx.transpose(), self.norm_param_y)
        return np.add(xIn, np.multiply(dx, dt))
    
    def stateTransitionVariance(self, xIn):
        xIn = np.array([xIn])
        x = self.normalize(xIn, self.norm_param_x)
        
        _ , var = self.gp.predict_noiseless(Xnew=[x]*self.n)

        var = self.denormalize(var.transpose(), self.norm_param_y)
        Q = np.diagflat(var)
        #TODO: is this correct or what should I do with dt?
        return Q
    

class GP_SSM_gpy_LVMOGP(GP_SSM_gpy_multiout):
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
    def __init__(self, dxData:np.ndarray, yData:np.ndarray, n:int, normalize=False, 
                 kern=GPy.kern.RBF
    ):
        super().__init__(dxData, yData, n, normalize, kern)
        
        D = n #   should be equal to y.shape[1] #output dimension
        Mr = D
        Mc = self.x_train.shape[0] #number of samples

        # WATCH OUT: latent dimension can have a tremendous impact on the performance (Qr=5 with threeTank guessed same Value for all states)
        Qr = 3

        self.gp = GPy.models.GPMultioutRegression(
            self.x_train,
            self.y_train,
            Xr_dim=Qr, 
            kernel_row=kern(Qr,ARD=True), #TODO what is ARD doing 
            num_inducing=(Mc,Mr),
            init='GP'
        )        

    def optimize(self, iterations=50, verbose=False):
        self.gp.optimize_auto(max_iters=iterations, verbose=verbose)
        
        
    def stateTransition(self, xIn, dt):
        xIn = np.array([xIn])
        x = self.normalize(xIn, self.norm_param_x)
            
        dx, _ = self.gp.predict_noiseless(x)
        
        dx = self.denormalize(dx, self.norm_param_y)
        return np.add(xIn, np.multiply(dx, dt))
    
    def stateTransitionVariance(self, xIn):
        xIn = np.array([xIn])
        x = self.normalize(xIn, self.norm_param_x)
        
        _ , var = self.gp.predict_noiseless(x)

        var = self.denormalize(var, self.norm_param_y)
        Q = np.diagflat(var)
        #TODO: is this correct or what should I do with dt?
        return Q
        

class GP_SSM_gpytorch_multitask(GP_SSM):
    '''
    This is a wrapper for a state space system for the MultitaskGPModel class in GPytorch
    which is an implementation of Multi-Task Gp Prediction from [Bonilla 2007]
    
    https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/index.html

    param dx: shape(D, n) where D is the number of samples and n is the output dimension
    param y: shape(D, n) where D is the number of samples and n is the output dimension
    '''
    def __init__(self, dxData:np.ndarray, yData:np.ndarray, n:int, normalize=False, 
                 kern=None, 
                 likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood,
    ):
        super().__init__(dxData, yData, n, normalize)
        
        self.x_train = torch.tensor(self.x_train).float()
        self.y_train = torch.tensor(self.y_train).float()
        
        self.likelihood = likelihood(num_tasks=self.n)
        self.gp = MultitaskGPModel(self.x_train , self.y_train , self.likelihood, num_tasks=self.n)
        
    '''
    TODO: this can be used general for all gpytorch models (move out of class)
    '''    
    def optimize(self, iterations=50, verbose=False):
        # Find optimal model hyperparameters
        self.gp .train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

        for i in range(iterations):
            optimizer.zero_grad()
            output = self.gp(self.x_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            if verbose: print('Iter %d/%d - Loss: %.3f' % (i + 1, iterations, loss.item()))
            optimizer.step()
            
        self.gp.eval()
        self.likelihood.eval()
 
    def stateTransition(self, xIn, dt):
        xIn = np.array([xIn])
        x = self.normalize(xIn, self.norm_param_x)
            
        #TODO dont use forward but single functions instead
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dx = self.gp.mean_module(torch.tensor(x).float()).numpy()
            # predictions = self.likelihood(self.gp(torch.tensor(x).float()))
            # dx = predictions.mean.numpy()
        
        dx = self.denormalize(dx, self.norm_param_y)
        return np.add(xIn, np.multiply(dx, dt))
    
    def stateTransitionVariance(self, xIn):
        xIn = np.array([xIn])
        x = self.normalize(xIn, self.norm_param_x)
        
        #TODO dont use forward but single functions instead
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            var = self.gp.covar_module(torch.tensor(x).float()).numpy()
            # predictions = self.likelihood(self.gp(torch.tensor(x).float()))
            # var = predictions.stddev.numpy()

        var = self.denormalize(var, self.norm_param_y)
        Q = np.diagflat(var)
        #TODO: is this correct or what should I do with dt?
        return var #TODO: this is not the rifht thing to do
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


'''
TODO: this is almost a duplicate from https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html
'''
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
import torchfilter
from torchfilter import types
import torch
import torch.nn as nn
from typing import Tuple, cast
from abc import abstractmethod
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from test_filters import _run_filter
from _linear_system_models import LinearDynamicsModel, LinearParticleFilterMeasurementModel
from _linear_system_fixtures import generated_data
from util import normalize_min_max_torch, denormalize_min_max
from GP_BF import MultitaskGPModel, BatchIndependentMultitaskGPModel
from threeTank import ThreeTank, parameter as param
from helper import createTrainingData

'''
TODO:
- generate data: Tuple[
        types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
    ], 
    Shapes of all inputs should be `(T, N, *)`.
X create dynamics_model equal to LinearDynamicsModel(),
X create measurement_model equal to LinearParticleFilterMeasurementModel(),

'''
dt=0.1
T = 100
x0 = np.zeros(3)

metaParams = [
    {'T':T, 'downsample':1},
    {'T':T, 'downsample':1}, 
]

metaParams2 = [
    {'T':T, 'downsample':10}, 
    #{'T':T, 'downsample':10}, 
]

param2 = param.copy()
param2['c13'] = param['c13'] * 4 
param2['c32'] = param['c32'] * 4
param2['c2R'] = param['c2R'] * 4
param2['u'] = 0

params = [
    param,
    param2
]
xD, yD, dxD, tsD = createTrainingData(ThreeTank, params, metaParams, 3, dt, x0, multipleSets=False, plot =False)
xD2, yD2, dxD2, tsD2 = createTrainingData(ThreeTank, [param], metaParams2, 3, dt, x0, multipleSets=False, plot =True)

xTest = torch.tensor(xD.transpose()).float()
yTest = torch.tensor(yD[0:3,:].transpose()).float() #TODO: adjustment needs to be automated
#uTest = torch.zeros_like(yTest)
uTest = torch.zeros(xTest.shape[0], 1, 1)

testShapeX = list(xTest.shape)
testShapeX.insert(1,1)
testShapeY = list(yTest.shape)
testShapeY.insert(1,1)
xTest = xTest.view(*testShapeX)
yTest = yTest.view(*testShapeY)

testData = (xTest, yTest, uTest)

xTrain = torch.tensor(xD2.transpose()).float()
dxTrain = torch.tensor(dxD2.transpose()).float()

sigma_y = 1e-7

# shape = list(tensor.shape)
# shape.insert(dim, 1)
# return tensor.view(*shape)

def test_particle_filter(generated_data):
    """Smoke test for particle filter."""

    gpModel = GpDynamicsModel(3, 1, xTrain, dxTrain, normalize=True)
    gpModel.optimize(verbose=True, iterations=50)
    estimate = _run_filter(
        torchfilter.filters.ParticleFilter(
            dynamics_model=gpModel,
            measurement_model=IdentityParticleFilterMeasurementModel(3, 3, (True, True, True), sigma_y), #TODO: adjustment needs to be automated
        ),
        generated_data,
    )

    x1 = estimate[:,:,0].numpy()
    x2 = estimate[:,:,1].numpy()
    x3 = estimate[:,:,2].numpy()
    plt.figure()
    plt.plot(tsD[:-1],x1, label = 'x1')
    plt.plot(tsD[:-1],x2, label = 'x2')
    plt.plot(tsD[:-1],x3, label = 'x3')

    plt.plot(tsD,yD[0,:], label = 'y1')
    plt.plot(tsD,yD[1,:], label = 'y2')
    plt.plot(tsD,yD[2,:], label = 'y3')
    plt.legend()
    plt.show()



#data = generated_data()

class IdentityKalmanFilterMeasurementModel(torchfilter.base.KalmanFilterMeasurementModel):
    """Kalman filter measurement model for our linear system. Maps states to
    (observation, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, state_dim:int, observation_dim:int, observable, sigma_y, trainable: bool = False):
        super().__init__(state_dim=state_dim, observation_dim=observation_dim)

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for

        #observable = tuple([True] * state_dim)

        # assert state_dim == observation_dim
        #self.C = torch.eye(state_dim)
        self.C = torch.zeros((observation_dim, state_dim))
        j = 0
        for i in range(len(observable)):
            if observable[i]:
                self.C[j,i] = 1
                j += 1
        self.R = torch.eye(observation_dim) * sigma_y

        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(
        self, *, states: types.StatesTorch
    ) -> Tuple[types.ObservationsNoDictTorch, types.ScaleTrilTorch]:
        """Observation model forward pass, over batch size `N`.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, state_dim)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing expected observations
            and cholesky decomposition of covariance.  Shape should be `(N, M)`.
        """
        # Check shape
        N = states.shape[0]
        assert states.shape == (N, self.state_dim)

        # Compute output

        observations = (self.C[None, :, :] @ states[:, :, None]).squeeze(-1)
        scale_tril = self.R[None, :, :].expand((N, self.observation_dim, self.observation_dim))

        # Add output bias if trainable
        if self.trainable:
            observations += self.output_bias

        # Compute/return predicted measurement and noise values
        return observations, scale_tril


class IdentityParticleFilterMeasurementModel(
    torchfilter.base.ParticleFilterMeasurementModelWrapper
):
    """Particle filter measurement model. Defined by wrapping our Kalman filter one.

    Distinction: the particle filter measurement model maps (state, observation) pairs
    to log-likelihoods (log particle weights), while the kalman filter measurement model
    maps states to (observation, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, state_dim, observation_dim, observable, sigma_y, trainable: bool = False):
        super().__init__(
            kalman_filter_measurement_model=IdentityKalmanFilterMeasurementModel(
                state_dim=state_dim, 
                observation_dim=observation_dim, 
                observable=observable,
                sigma_y=sigma_y,
                trainable=trainable
            )
        )


class GpDynamicsModel(torchfilter.base.DynamicsModel):
    """Forward model for our GP system. Maps (initial_states, controls) pairs to
    (predicted_state, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, 
                 state_dim:int, 
                 dt:int,
                 xData, 
                 dxData, 
                 kern=None, 
                 likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood,
                 model= BatchIndependentMultitaskGPModel, #MultitaskGPModel,
                 normalize=False, 
                 trainable: bool = False
    ):
        super().__init__(state_dim=state_dim)

        self.norm_param_x = None
        self.norm_param_dx = None
        self.normFct  = None
        self.denormFct = None

        if normalize:
            self.normFct = normalize_min_max_torch
            self.denormFct = denormalize_min_max
        
            self.x_train, self.norm_param_x =  self.normFct(xData)
            self.dx_train, self.norm_param_dx =  self.normFct(dxData)
        else:
            self.x_train = xData
            self.dx_train = dxData

        self.dt = dt

        self.likelihood = likelihood(num_tasks=self.state_dim)
        self.gp = model(self.x_train , self.dx_train , self.likelihood, num_tasks=self.state_dim)

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1]))

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
            loss = -mll(output, self.dx_train)
            loss.backward()
            if verbose: print('Iter %d/%d - Loss: %.3f' % (i + 1, iterations, loss.item()))
            optimizer.step()
            
        self.gp.eval()
        self.likelihood.eval()
    
    def normalize(self, x, param):
        if self.normFct:
            x, _ = self.normFct(x, *param)
        return x
    
    def denormalize(self, x, param):
        if self.denormFct:
            x = self.denormFct(x, *param)
        return x

    def forward(
        self,
        *,
        initial_states: types.StatesTorch,
        controls: types.ControlsTorch,
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
        """Forward step for a discrete linear dynamical system.

        Args:
            initial_states (torch.Tensor): Initial states of our system.
            controls (dict or torch.Tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted states & uncertainties.
                - States should have shape `(N, state_dim).`
                - Uncertainties should be lower triangular, and should have shape
                `(N, state_dim, state_dim).`
        """

        # Controls should be tensor, not dictionary
        assert isinstance(controls, torch.Tensor)
        controls = cast(torch.Tensor, controls)
        # Check shapes
        N, state_dim = initial_states.shape
        N_alt, control_dim = controls.shape
        assert self.state_dim == state_dim
        assert N == N_alt

        # my code
        #x = self.normalize(initial_states[:,:,None], self.norm_param_x)
        x = self.normalize(initial_states, self.norm_param_x)
            
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #dx = self.gp.mean_module(torch.tensor(x).float())
            predictions = self.likelihood(self.gp(x))
            dx = predictions.mean
            var = predictions.stddev.diag_embed()
        
        # if dx.shape[1] == 1:
        #     dx = dx.transpose()

        # if var.shape[1] == 1:
        #     var =torch.diagflat(var)

        var = self.denormalize(var, self.norm_param_dx)
        dx = self.denormalize(dx, self.norm_param_dx)
        #return xIn + dx * dt
        predicted_states = initial_states + dx.squeeze(-1)* self.dt
        

        #-------------------------------

        # Compute/return states and noise values
        # predicted_states = (A[None, :, :] @ initial_states[:, :, None]).squeeze(-1) + (
        #     B[None, :, :] @ controls[:, :, None]
        # ).squeeze(-1)

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias

        #return predicted_states, var.tril()
        varS = var @ var.mT + 1e-7
        return predicted_states, varS.cholesky()
        return predicted_states, var[None, :, :].expand((N, state_dim, state_dim))
    

'''
class ThreeTankModel(torchfilter.base.DynamicsModel):
    """Forward model for our linear system. Maps (initial_states, controls) pairs to
    (predicted_state, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """
    state_dim = 3

    def __init__(self, trainable: bool = False):
        super().__init__(state_dim=self.state_dim)

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(
        self,
        *,
        initial_states: types.StatesTorch,
        controls: types.ControlsTorch,
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
        """Forward step for a discrete linear dynamical system.

        Args:
            initial_states (torch.Tensor): Initial states of our system.
            controls (dict or torch.Tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted states & uncertainties.
                - States should have shape `(N, state_dim).`
                - Uncertainties should be lower triangular, and should have shape
                `(N, state_dim, state_dim).`
        """

        # Controls should be tensor, not dictionary
        assert isinstance(controls, torch.Tensor)
        controls = cast(torch.Tensor, controls)
        # Check shapes
        N, state_dim = initial_states.shape
        N_alt, control_dim = controls.shape
        assert state_dim == self.state_dim
        #assert A.shape == (state_dim, state_dim)
        assert N == N_alt

        # Compute/return states and noise values
        predicted_states = (A[None, :, :] @ initial_states[:, :, None]).squeeze(-1) + (
            B[None, :, :] @ controls[:, :, None]
        ).squeeze(-1)

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias

        return predicted_states, Q_tril[None, :, :].expand((N, state_dim, state_dim))

'''

test_particle_filter(testData)
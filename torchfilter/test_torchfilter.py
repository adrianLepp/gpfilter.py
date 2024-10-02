import torchfilter
from torchfilter import types
import torch
import torch.nn as nn
from typing import Tuple, cast, Optional, List
from abc import abstractmethod
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import fannypack
from overrides import overrides

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
T = 50
x0 = np.zeros(3)

metaParams = [
    {'T':T, 'downsample':1},
    {'T':T, 'downsample':1}, 
]

metaParams2 = [
    {'T':T, 'downsample':1}, 
    {'T':T, 'downsample':1}, 
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

xD2, yD2, dxD2, tsD2 = createTrainingData(ThreeTank, [param, param2] , metaParams2, 3, dt, x0, multipleSets=True, plot =True)

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

xTrain1 = torch.tensor(xD2[0].transpose()).float()
xTrain2 = torch.tensor(xD2[1].transpose()).float()
dxTrain1 = torch.tensor(dxD2[0].transpose()).float()
dxTrain2 = torch.tensor(dxD2[1].transpose()).float()

sigma_y = 1e-5
# shape = list(tensor.shape)
# shape.insert(dim, 1)
# return tensor.view(*shape)

def _run_imm_filter(
    filter_model: torchfilter.base.Filter,
    data: Tuple[
        types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
    ],
    initialize_beliefs: bool = True,
) -> torch.Tensor:
    """Helper for running a filter and returning estimated states.

    Args:
        filter_model (torchfilter.base.Filter): Filter to run.
        data (Tuple[
            types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
        ]): Data to run on. Shapes of all inputs should be `(T, N, *)`.

    Returns:
        torch.Tensor: Estimated states. Shape should be `(T - 1, N, state_dim)`.
    """

    # Get data
    states, observations, controls = data
    T, N, state_dim = states.shape

    # Initialize the filter belief to match the first timestep
    if initialize_beliefs:
        filter_model.initialize_beliefs(
            mean=states[0],
            covariance=torch.zeros(size=(N, state_dim, state_dim))
            + torch.eye(state_dim)[None, :, :] * 1e-7,
        )

    # Run the filter on the remaining `T - 1` timesteps
    estimated_states, estimated_modes = filter_model.forward_loop(
        observations=observations[1:], controls=controls[1:]
    )

    # Check output and return
    assert estimated_states.shape == (T - 1, N, state_dim)
    return estimated_states, estimated_modes

def test_particle_filter(generated_data):
    """Smoke test for particle filter."""

    gpModel1 = GpDynamicsModel(3, 1, xTrain1, dxTrain1, normalize=True)
    gpModel1.optimize(verbose=True, iterations=50)

    gpModel2 = GpDynamicsModel(3, 1, xTrain2, dxTrain2, normalize=True)
    gpModel2.optimize(verbose=True, iterations=50)

    estimates, modes = _run_imm_filter(
        IMMParticleFilter(
            dynamics_models=[gpModel1, gpModel2],
            measurement_model=IdentityParticleFilterMeasurementModel(3, 3, (True, True, True), sigma_y), #TODO: adjustment needs to be automated
            mu = [0.9, 0.1],
            Pi = torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            state_dim= 3,
            num_particles=100,
            #estimation_method='argmax'
        ),
        generated_data,
    )

    # estimate = _run_filter(
    #     torchfilter.filters.ParticleFilter(
    #         dynamics_model=gpModel1,
    #         measurement_model=IdentityParticleFilterMeasurementModel(3, 1, (True, False, False), sigma_y), #TODO: adjustment needs to be automated
    #     ),
    #     generated_data,
    # )

    x1 = estimates[:,:,0].numpy()
    x2 = estimates[:,:,1].numpy()
    x3 = estimates[:,:,2].numpy()
    plt.figure()
    plt.plot(tsD[:-1],x1, label = 'x1')
    plt.plot(tsD[:-1],x2, label = 'x2')
    plt.plot(tsD[:-1],x3, label = 'x3')

    plt.plot(tsD,yD[0,:], label = 'y1')
    plt.plot(tsD,yD[1,:], label = 'y2')
    plt.plot(tsD,yD[2,:], label = 'y3')
    plt.legend()

    plt.figure()
    plt.plot(tsD[:-1],modes[:,:,0].numpy(), label = 'mode1')
    plt.plot(tsD[:-1],modes[:,:,1].numpy(), label = 'mode2')
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
                 model= MultitaskGPModel, # BatchIndependentMultitaskGPModel, #
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
        return predicted_states, torch.linalg.cholesky(varS)
        return predicted_states, var[None, :, :].expand((N, state_dim, state_dim))

class IMMParticleFilter(torchfilter.base.Filter):
    """Generic differentiable particle filter."""

    def __init__(
        self,
        *,
        dynamics_models: List[torchfilter.base.DynamicsModel],
        measurement_model: torchfilter.base.ParticleFilterMeasurementModel,
        mu: List[float],
        Pi: torch.Tensor,
        state_dim: int,
        num_particles: int = 100,
        resample: Optional[bool] = None,
        soft_resample_alpha: float = 1.0,
        estimation_method: str = "weighted_average",
    ):
        
        # Check submodule consistency
        #assert isinstance(dynamics_model, DynamicsModel)
        assert isinstance(measurement_model, torchfilter.base.ParticleFilterMeasurementModel)
        if len(dynamics_models) < 2:
            raise ValueError('filters must contain at least two filters')

        # Initialize state dimension
        state_dim = state_dim
        super().__init__(state_dim=state_dim)

        # Assign submodules
        self.dynamics_models = dynamics_models
        """torchfilter.base.DynamicsModel: Forward model."""
        self.measurement_model = measurement_model
        """torchfilter.base.ParticleFilterMeasurementModel: Observation model."""

        # Settings                
        self.Pi = Pi
        self.mu = mu
        self.num_modes = len(dynamics_models)
        self.num_particles = num_particles
        """int: Number of particles to represent our belief distribution.
        Defaults to 100."""
        self.resample = resample
        """bool: If True, we resample particles & normalize weights at each
        timestep. If unset (None), we automatically turn resampling on in eval mode
        and off in train mode."""

        self.soft_resample_alpha = soft_resample_alpha
        """float: Tunable constant for differentiable resampling, as described
        by Karkus et al. in "Particle Filter Networks with Application to Visual
        Localization": https://arxiv.org/abs/1805.08975
        Defaults to 1.0 (disabled)."""

        assert estimation_method in ("weighted_average", "argmax")
        self.estimation_method = estimation_method
        """str: Method of producing state estimates. Options include:
        - 'weighted_average': average of particles weighted by their weights.
        - 'argmax': state of highest weighted particle.
        """

        # "Hidden state" tensors
        self.particle_states: torch.Tensor
        """torch.Tensor: Discrete particles representing our current belief
        distribution. Shape should be `(N, M, state_dim)`.
        """
        self.particle_log_weights: torch.Tensor
        """torch.Tensor: Weights corresponding to each particle, stored as
        log-likelihoods. Shape should be `(N, M)`.
        """
        self._initialized = False

    @overrides
    def initialize_beliefs(
        self, *, mean: types.StatesTorch, covariance: types.CovarianceTorch
    ) -> None:
        """Populates initial particles, which will be normally distributed.

        Args:
            mean (torch.Tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.Tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)
        self.num_modes = self.num_modes
        self.num_particles = self.num_particles

        # Sample particles
        self.particle_states = (
            torch.distributions.MultivariateNormal(mean, covariance)
            .sample((self.num_particles,self.num_modes))
            .transpose(0, 2)
        )
        assert self.particle_states.shape == (N, self.num_modes, self.num_particles, self.state_dim)

        self.particle_log_weights = torch.zeros((N, self.num_modes, self.num_particles), dtype=torch.float32)
        for j in range(self.num_modes):
            self.particle_log_weights[:,j,:] =  np.log(self.mu[j]/self.num_particles)

        # Normalize weights #TODO: weights need to add up to one over all modes
        # self.particle_log_weights = self.particle_states.new_full(
        #     #(N, self.num_modes,  self.num_particles), float(-np.log(self.num_particles * self.num_modes, dtype=np.float32))
        #     (N, self.num_modes,  self.num_particles), float(np.log(self.mu[], self.num_particles, dtype=np.float32))
        # )
        assert self.particle_log_weights.shape == (N, self.num_modes, self.num_particles)

        # Set initialized flag
        self._initialized = True
    
    @overrides
    def forward(
        self,
        *,
        observations: types.ObservationsTorch,
        controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """Interacting Multiple Model Particle filter forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`.
            controls (dict or torch.Tensor): control inputs. should be either a
                dict of tensors or tensor of shape `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """

        # Make sure our particle filter's been initialized
        assert self._initialized, "Particle filter not initialized!"

        # Get our batch size (N), mode number (q) current particle count (M), & state dimension
        N, q, M, state_dim = self.particle_states.shape
        assert state_dim == self.state_dim
        assert q == self.num_modes
        assert len(fannypack.utils.SliceWrapper(controls)) == N

        # Decide whether or not we're resampling
        resample = self.resample
        if resample is None:
            # If not explicitly set, we disable resampling in train mode (to allow
            # gradients to propagate through time) and enable in eval mode (to prevent
            # particle deprivation)
            resample = not self.training

        # If we're not resampling and our current particle count doesn't match
        # our desired particle count, we need to either expand or contract our
        # particle set
        if not resample and self.num_particles != M:
            print.warn("expansio / contraction of particle set is not implemented!")
            #TODO: implement this. Whatever it does


        # Propagate particles through our dynamics model
        # A bit of extra effort is required for the extra particle dimension
        # > For our states, we flatten along the N/M axes
        # > For our controls, we repeat each one `M` times, if M=3:
        #       [u0 u1 u2] should become [u0 u0 u0 u1 u1 u1 u2 u2 u2]
        #
        # Currently each of the M particles within a "sample" get the same action, but
        # we could also add noise in the action space (a la Jonschkowski et al. 2018)
        reshaped_states = self.particle_states.reshape(-1, self.num_particles, self.state_dim) #reduced to  q x M x state_dim
        reshaped_controls = fannypack.utils.SliceWrapper(controls).map(
            lambda tensor: torch.repeat_interleave(tensor, repeats=M, dim=0)
        ) #map controls to every particle: 100x1

        # 1. interaction step
        mPostK = self.particle_log_weights[:, :, :].exp().sum(dim=2)

        mPrioKK =   mPostK @ self.Pi

        # create particle set with self.num_particles*self.num_modes particles per Mode 
        # to approximate prior density after mode change
        xResampling = reshaped_states.reshape(self.num_particles * self.num_modes, self.state_dim)

        wResampling = torch.zeros(self.num_modes, self.num_particles*self.num_modes)
        for j in range(self.num_modes):
            for i in range(self.num_modes):
                for l in range(self.num_particles):
                    wResampling[j, l+(i)*self.num_particles] = self.Pi[i,j]*self.particle_log_weights[:,i,l].exp()  / mPrioKK[:,j] 


        # Reduce number of particles to self.num_particles per Mode by Resampling
        xPrio = torch.zeros(N,self.num_modes,self.num_particles,self.state_dim)
        wPrio = torch.zeros(N, self.num_modes,self.num_particles)
        for j in range(self.num_modes):
            if mPrioKK[:,j] <= self.num_modes/(self.num_particles*10): # if prior mode probabilty to low, take over posterior k-1
                xPrio[:,j,:,:] = reshaped_states[:,j,:,:]
            else:
                sumW = wResampling[j,:].sum(dim=0)
                for l in range(self.num_particles):
                    #pos = find(torch.rand <= torch.cumsum(wResampling[j,:]),1)

                    #pos = torch.nonzero(torch.cumsum(wResampling[j, :], dim=0) < torch.rand(1) * sumW)[0].item()
                    pos = torch.nonzero(torch.cumsum(wResampling[j, :], dim=0) >= torch.rand(1) * sumW)[0].item()
                    xPrio[:,j,l,:] = xResampling[pos,:]       
                #
            #
            # new prior weights
            wPrio[:,j,:] = torch.log(mPrioKK[:,j] / self.num_particles)

        reshaped_states = xPrio.reshape(-1, self.num_particles, self.state_dim)

        predicted_states = torch.zeros(N,self.num_modes,self.num_particles,self.state_dim)

        # 2. prediction step
        for j in range(self.num_modes):
            predicted_states[:,j,:,:], scale_trils = self.dynamics_models[j](
                initial_states=reshaped_states[j,:,:], controls=reshaped_controls
            )
        
        #TODO: is this resampling here at the right place? What does it actually do? 
        self.particle_states = (
            torch.distributions.MultivariateNormal(
                loc=predicted_states, scale_tril=scale_trils
            )
            .rsample()  # Note that we use `rsample` to make sampling differentiable
            .view(N, q, M, self.state_dim)
        )
        assert self.particle_states.shape == (N, q, M, self.state_dim)


        # 3. correction step Re-weight particles using observations
        weightCorrect = torch.zeros(N, self.num_modes, self.num_particles)


        for j in range(self.num_modes):
            weightCorrect[:,j,:] = self.measurement_model(
                states=self.particle_states[:,j,:,:],
                observations=observations,
            )
            # self.particle_log_weights[:,j,:] = wPrio[:,j,:] + self.measurement_model(
            #     states=self.particle_states[:,j,:,:],
            #     observations=observations,
            # )
        self.particle_log_weights = wPrio + weightCorrect

        # Normalize particle weights to sum to 1.0
        self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
            self.particle_log_weights, dim=(1,2), keepdim=True #TODO dim=2? 
        ) #TODO normalization is not correct. accros all modes should sum to 1

        # Compute output
        state_estimates: types.StatesTorch
        
        mPost = self.particle_log_weights.exp().sum(dim=2)
        #print('mode Probability', mPost)
        #xEstM = torch.zeros(N, self.num_modes, self.state_dim)

        if self.estimation_method == "weighted_average":
            xEstM = (self.particle_log_weights[:, :, :, np.newaxis].exp()* self.particle_states[:, :, : ,: ]).sum(dim=2)
    
            # for j in range(self.num_modes):
            #     xEstM[:,j,:] = torch.sum(
            #         torch.exp(self.particle_log_weights[:, j, :, np.newaxis])
            #         * self.particle_states[:, j, : ,: ],
            #         dim=1,
            #     )
            state_estimates = xEstM.sum(dim=1)
            modeEstimates = self.particle_log_weights.exp().sum(dim=2)

        elif self.estimation_method == "argmax":
            best_indices = torch.argmax(self.particle_log_weights, dim=2)
            # state_estimates = torch.gather(
            #     self.particle_states, dim=2, index=best_indices
            # )
            state_estimates = self.particle_states[:,0,best_indices[0,0],:]
        else:
            assert False, "Unsupported estimation method!"

        # Resampling
        if resample:
            self._resample()

        # Post-condition :)
        assert state_estimates.shape == (N, state_dim)
        assert self.particle_states.shape == (N, self.num_modes, self.num_particles, state_dim)
        assert self.particle_log_weights.shape == (N, self.num_modes, self.num_particles)

        return state_estimates, modeEstimates
    
    def forward_loop(
        self, *, observations: types.ObservationsTorch, controls: types.ControlsTorch
    ) -> types.StatesTorch:
        """Filtering forward pass, over sequence length `T` and batch size `N`.
        By default, this is implemented by iteratively calling `forward()`.

        To inject code between timesteps (for example, to inspect hidden state),
        use `register_forward_hook()`.

        Args:
            observations (dict or torch.Tensor): observation inputs. Should be
                either a dict of tensors or tensor of size `(T, N, ...)`.
            controls (dict or torch.Tensor): control inputs. Should be either a
                dict of tensors or tensor of size `(T, N, ...)`.

        Returns:
            torch.Tensor: Predicted states at each timestep. Shape should be
            `(T, N, state_dim).`
        """

        # Wrap our observation and control inputs
        #
        # If either of our inputs are dictionaries, this provides a tensor-like
        # interface for slicing, accessing shape, etc
        observations_wrapped = fannypack.utils.SliceWrapper(observations)
        controls_wrapped = fannypack.utils.SliceWrapper(controls)

        # Get sequence length (T), batch size (N)
        T, N = controls_wrapped.shape[:2]
        assert observations_wrapped.shape[:2] == (T, N)

        # Filtering forward pass
        # We treat t = 0 as a special case to make it easier to create state_predictions
        # tensor on the correct device
        t = 0
        current_prediction, current_mode = self(
            observations=observations_wrapped[t], controls=controls_wrapped[t]
        )
        state_predictions = current_prediction.new_zeros((T, N, self.state_dim))
        mode_predictions = torch.zeros((T, N, self.num_modes))
        assert current_prediction.shape == (N, self.state_dim)
        state_predictions[t] = current_prediction
        mode_predictions[t] = current_mode

        for t in range(1, T):
            # Compute state prediction for a single timestep
            # We use __call__ to make sure hooks are dispatched correctly
            current_prediction, current_mode_prediction = self(
                observations=observations_wrapped[t], controls=controls_wrapped[t]
            )

            # Validate & add to output
            assert current_prediction.shape == (N, self.state_dim)
            state_predictions[t] = current_prediction
            mode_predictions[t] = current_mode_prediction

        # Return state predictions
        return state_predictions, mode_predictions

test_particle_filter(testData)


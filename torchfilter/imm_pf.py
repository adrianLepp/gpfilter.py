import torchfilter
from torchfilter import types
import torch
from typing import Optional, List
import numpy as np
import fannypack
from overrides import overrides

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
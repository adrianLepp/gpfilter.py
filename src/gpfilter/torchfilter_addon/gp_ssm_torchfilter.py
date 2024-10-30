import torchfilter
from torchfilter import types
import torch
import torch.nn as nn
from typing import Tuple, cast
import gpytorch
gpytorch.settings.debug(False)
# --------------------------------------------------------------------------------
from gpfilter.gp import MultitaskGPModel, BatchIndependentMultitaskGPModel, ConvolvedGPModel
from gpfilter.utils import normalize_min_max_torch, denormalize_min_max

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
                 sigma_x, 
                 kern = None, 
                 likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood,
                 model =  BatchIndependentMultitaskGPModel, 
                 normalize = False, 
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
        self.Q = torch.eye(state_dim) * sigma_x

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

        x = self.normalize(initial_states, self.norm_param_x)
            
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            predictions = self.likelihood(self.gp(x))
            dx = predictions.mean
            var = predictions.stddev.diag_embed()

        var = self.denormalize(var, self.norm_param_dx)
        dx = self.denormalize(dx, self.norm_param_dx)
        predicted_states = initial_states + dx.squeeze(-1)* self.dt

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias

        #return predicted_states, var.tril()
        varS = var @ var.mT + 1e-7
        return predicted_states, torch.linalg.cholesky(varS)
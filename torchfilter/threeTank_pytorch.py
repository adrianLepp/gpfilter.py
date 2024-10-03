import torchfilter
from torchfilter import types
import torch
import torch.nn as nn
from typing import Tuple, cast

parameter = {
    'u': 1.2371e-4,
    'c13': 2.5046e-5,
    'c32': 2.5046e-5,
    'c2R': 1.9988e-5,
    'g': 9.81 ,
    'A': 0.0154,
    'sigmaX': 1e-6,
    'sigmaY': 5e-4,
}

class ThreeTankDynamicsModel(torchfilter.base.DynamicsModel):
    """Forward model for our GP system. Maps (initial_states, controls) pairs to
    (predicted_state, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, 
                 state_dim:int, 
                 dt:int,
                 param=parameter,
                 trainable: bool = False
    ):
        super().__init__(state_dim=state_dim)

        self.u = param['u']
        self.c13 = param['c13']
        self.c32 = param['c32']
        self.c2R = param['c2R']
        self.A = param['A']
        self.g = param['g']

        self.sigmaX = param['sigmaX']
        self.sigmaY = param['sigmaY']

        self.dt = dt
        self.Q = torch.eye(state_dim) * self.sigmaX

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
        assert self.state_dim == state_dim
        assert N == N_alt
            
        dx = self.stateTransition(initial_states)
        predicted_states = initial_states + dx * self.dt

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias
    
        return predicted_states, self.Q[None, :, :].expand((N, state_dim, state_dim))
    
    def stateTransition(self, x:types.StatesTorch):
        shape = x.shape
        x = torch.clamp(x, min=0)

        dx1 = 1/self.A*(self.u-self.c13*torch.sign(x[...,0]-x[...,2])*torch.sqrt(2*self.g*abs(x[...,0]-x[...,2])))
        dx2 = 1/self.A*(self.c32*torch.sign(x[...,2]-x[...,1])*torch.sqrt(2*self.g*abs(x[...,2]-x[...,1]))-self.c2R*torch.sqrt(2*self.g*abs(x[...,1])))
        dx3 = 1/self.A*(self.c13*torch.sign(x[...,0]-x[...,2])*torch.sqrt(2*self.g*abs(x[...,0]-x[...,2]))-self.c32*torch.sign(x[...,2]-x[...,1])*torch.sqrt(2*self.g*abs(x[...,2]-x[...,1])))

        dx = torch.cat((
            dx1.unsqueeze(-1),
            dx2.unsqueeze(-1),
            dx3.unsqueeze(-1)
            ),-1)

        assert dx.shape == shape
        return dx
        # noise = torch.randn(size=shape) * self.sigmaX
        # dx_noisy = dx + noise
        
        return dx_noisy    
    
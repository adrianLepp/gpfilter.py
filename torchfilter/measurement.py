

import torchfilter
from torchfilter import types
import torch
import torch.nn as nn
from typing import Tuple

class IdentityKalmanFilterMeasurementModel(torchfilter.base.KalmanFilterMeasurementModel):
    """Kalman filter measurement model where states are directly measured. Maps states to
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

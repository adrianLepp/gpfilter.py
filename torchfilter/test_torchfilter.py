import torchfilter
from torchfilter import types
import torch
import torch.nn as nn
from typing import Tuple, cast, Optional, List
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import fannypack
from overrides import overrides

from threeTank import ThreeTank, parameter as param
from helper import createTrainingData
from measurement import IdentityParticleFilterMeasurementModel
from gpModel import GpDynamicsModel
from imm_pf import IMMParticleFilter

dt=0.1
T = 50
x0 = np.zeros(3)

metaParams = [
    {'T':T, 'downsample':1},
    {'T':T, 'downsample':1}, 
]

metaParams2 = [
    {'T':T, 'downsample':10}, 
    {'T':T, 'downsample':10}, 
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
yTest = torch.tensor(yD[0:2,:].transpose()).float() #TODO: adjustment needs to be automated
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

sigma_y = 1e-7
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
            measurement_model=IdentityParticleFilterMeasurementModel(3, 2, (True, True, False), sigma_y), #TODO: adjustment needs to be automated
            mu = [0.5, 0.5],
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

test_particle_filter(testData)


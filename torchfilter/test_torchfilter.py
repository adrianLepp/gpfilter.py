import torchfilter
from torchfilter import types
import torch
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from threeTank import ThreeTank, parameter as param
from helper import createTrainingData
from measurement import IdentityParticleFilterMeasurementModel
from gp_ssm import GpDynamicsModel
from threeTank_pytorch import ThreeTankDynamicsModel
from imm_pf import IMMParticleFilter
from test_filters import _run_filter

from GP_BF import GP_SSM_gpy_multiout, GP_SSM_gpy_LVMOGP, GP_SSM_gpytorch_multitask, BatchIndependentMultitaskGPModel


simCounter = 11

verbose = True
MULTI_MODEL = True
GP = True
SAVE = False

NORMALIZE = True
OPTIM_STEPS = 1

# -----------------------------------------------------------------------------
# model settings
#------------------------------------------------------------------------------

gp_model = GP_SSM_gpytorch_multitask


stateN = 3
measN = 1
x0 = np.zeros(stateN) # initial state
dt = 0.1
T = 50

#PF
sigma_y = 1e-2
sigma_x = 1e-7
S = 100

#IMMM
modeN = 2
mu = [0.9, 0.1] 
trans = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

param2 = param.copy()
param2['c13'] = param['c13'] * 4 
param2['c32'] = param['c32'] * 4
param2['c2R'] = param['c2R'] * 4
param2['u'] = 0


# -----------------------------------------------------------------------------
# create training data
#------------------------------------------------------------------------------

metaParams = [
    {'T':T, 'downsample':10}, 
    {'T':T, 'downsample':10}
]

params = [
    param, 
    param2
]

#if GP:
xD, yD, dxD, tsD = createTrainingData(ThreeTank, params, metaParams, stateN, dt, x0, multipleSets=MULTI_MODEL)


# -----------------------------------------------------------------------------
# create test data
#------------------------------------------------------------------------------

metaParamT = [
    {'T':T, 'downsample':1}, 
    {'T':T, 'downsample':1}
]

paramT = [
    param, 
    param2
]

xT, yT, dxT, tsT = createTrainingData(ThreeTank, paramT, metaParamT, stateN, dt, x0, multipleSets=False, plot =False)

xTest = torch.tensor(xT.transpose()).float()
yTest = torch.tensor(yT[0:1,:].transpose()).float() #TODO: adjustment needs to be automated
uTest = torch.zeros(xTest.shape[0], 1, 1)

testShapeX = list(xTest.shape)
testShapeX.insert(1,1)
testShapeY = list(yTest.shape)
testShapeY.insert(1,1)
xTest = xTest.view(*testShapeX)
yTest = yTest.view(*testShapeY)

testData = (xTest, yTest, uTest)



#param['sigmaX'] = 1e-4
# param['sigmaY'] = 0

# metaParams = [
#     {'T':T, 'downsample':1},
#     {'T':T, 'downsample':1}, 
# ]

# metaParams2 = [
#     {'T':50, 'downsample':10}, 
#     {'T':50, 'downsample':10}, 
# ]

# param2 = param.copy()
# param2['c13'] = param['c13'] * 4 
# param2['c32'] = param['c32'] * 4
# param2['c2R'] = param['c2R'] * 4
# param2['u'] = 0

# params = [
#     param,
#     param2
# ]
# xD, yD, dxD, tsD = createTrainingData(ThreeTank, params, metaParams, 3, dt, x0, multipleSets=False, plot =False)

# xD2, yD2, dxD2, tsD2 = createTrainingData(ThreeTank, [param, param2] , metaParams2, 3, dt, x0, multipleSets=True, plot =False)

# xTest = torch.tensor(xD.transpose()).float()
# yTest = torch.tensor(yD[0:1,:].transpose()).float() #TODO: adjustment needs to be automated
# #uTest = torch.zeros_like(yTest)
# uTest = torch.zeros(xTest.shape[0], 1, 1)

# testShapeX = list(xTest.shape)
# testShapeX.insert(1,1)
# testShapeY = list(yTest.shape)
# testShapeY.insert(1,1)
# xTest = xTest.view(*testShapeX)
# yTest = yTest.view(*testShapeY)

# testData = (xTest, yTest, uTest)


# xTrain = torch.tensor(xD2.transpose()).float()
# dxTrain = torch.tensor(dxD2.transpose()).float()
# xTrain1 = torch.tensor(xD2[0].transpose()).float()
# xTrain2 = torch.tensor(xD2[1].transpose()).float()
# dxTrain1 = torch.tensor(dxD2[0].transpose()).float()
# dxTrain2 = torch.tensor(dxD2[1].transpose()).float()

# sigma_y = 1e-2
# sigma_x = 1e-7
# shape = list(tensor.shape)
# shape.insert(dim, 1)
# return tensor.view(*shape)

def createFilter(stateN:int, measN:int, modeN: int, dt:float, xD, dxD, sigma_x:float, sigma_y:float, param, mu, trans_p, S:int ):
# init variables for IMM-GP-UKF

    meas_model:torchfilter.base.ParticleFilterMeasurementModel = IdentityParticleFilterMeasurementModel(stateN, measN, (True, False, False), sigma_y) #TODO: adjustment needs to be automated

    if MULTI_MODEL:
        models = []
        if GP:
            for i in range(modeN):
                gpModel = GpDynamicsModel(stateN, 1, torch.tensor(xD[i].transpose()).float(), torch.tensor(dxD[i].transpose()).float(), sigma_x, normalize=NORMALIZE)
                gpModel.optimize(verbose=False, iterations=OPTIM_STEPS)
                models.append(gpModel)
        else:
            for i in range(modeN):
                model =  ThreeTankDynamicsModel(stateN, dt, param[i])
                models.append(model)

        imm: torchfilter.base.Filter = IMMParticleFilter(
            dynamics_models=models,
            measurement_model= meas_model,
            mu = mu,
            Pi = trans_p,
            state_dim= stateN,
            num_particles=S,
        )

        return imm

    else:
        gpModel = GpDynamicsModel(stateN, 1, torch.tensor(xD.transpose()).float(), torch.tensor(dxD.transpose()).float(), sigma_x, normalize=NORMALIZE)
        gpModel.optimize(OPTIM_STEPS, verbose)
        pf = torchfilter.filters.ParticleFilter(
            dynamics_model=gpModel,
            measurement_model=meas_model
        )
        return pf

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

def test_particle_filter(filter, generated_data, multiModel:bool):
    if multiModel:
        estimates, modes = _run_imm_filter(filter, generated_data)
    else:
        estimates = _run_filter(filter, generated_data)
        modes = None

    true_states = generated_data[0]

    if modes is not None:
        modes = modes.squeeze().numpy().transpose()

    plotResults(tsT[:-1], true_states.squeeze().numpy().transpose()[:,:-1], estimates.squeeze().numpy().transpose(), muValues=modes)


    # x1 = estimates[:,:,0].numpy()
    # x2 = estimates[:,:,1].numpy()
    # x3 = estimates[:,:,2].numpy()
    # plt.figure()
    # plt.plot(tsD[:-1],x1, label = 'x1')
    # plt.plot(tsD[:-1],x2, label = 'x2')
    # plt.plot(tsD[:-1],x3, label = 'x3')

    # plt.plot(tsD,yD[0,:], label = 'y1')
    # plt.plot(tsD,yD[1,:], label = 'y2')
    # plt.plot(tsD,yD[2,:], label = 'y3')
    # plt.legend()

    # plt.figure()
    # plt.plot(tsD[:-1],modes[:,:,0].numpy(), label = 'mode1')
    # plt.plot(tsD[:-1],modes[:,:,1].numpy(), label = 'mode2')
    # plt.legend()

    # plt.show()


def plotResults(
        time, 
        zValues, 
        xValues, 
        errorValues = None, 
        varianceValues = None, 
        likelihoodValues = None, 
        muValues = None, 
        folder = None, 
        simName = None,
        lower_bound = None, 
        upper_bound = None, 
    ):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(stateN):
        ax[0,0].plot(time, zValues[i, :], label='$z_' + str(i) + '$')
        ax[0,0].plot(time, xValues[i, :], '--', label='$x_' + str(i) + '$')
        if lower_bound is not None and upper_bound is not None:
            ax[0,0].fill_between(time, lower_bound[i, :], upper_bound[i, :], color='gray', alpha=0.5, label='$95\%$ CI' if i == 0 else "")
    ax[0,0].legend()

    if errorValues is not None:
        for i in range(stateN):
            ax[0,1].plot(time, errorValues[i, :], label='$error_' + str(i) + '$')
        ax[0,1].legend()

    if likelihoodValues is not None:
        ax[1,0].plot(time, likelihoodValues[0,:], label='likelihood')
        ax[1,0].legend()

    if MULTI_MODEL and muValues is not None:
        #fig2, ax[0,1] = plt.subplots(1, 1, figsize=set_size(textWidth, 0.5,(1,1)))
        for i in range(modeN):
            ax[1,1].plot(time, muValues[i,:], label='$mu^' + str(i) + '$')
        ax[1,1].legend()
        #fig2.savefig('../gaussianProcess.tex/img/modeEstimation.pdf', format='pdf', bbox_inches='tight')

    # save data
    if SAVE and folder is not None and simName is not None:
        df_state = pd.DataFrame(data=xValues.transpose(), columns=['x' + str(i) for i in range(stateN)])
        df_error = pd.DataFrame(data=errorValues.transpose(), columns=['error' + str(i) for i in range(stateN)])
        df_variance = pd.DataFrame(data=varianceValues.transpose(), columns=['variance' + str(i) for i in range(stateN)])
        df_time = pd.DataFrame(data=time, columns=['time'])
        df_measurement = pd.DataFrame(data=zValues.transpose(), columns=['z' + str(i) for i in range(stateN)])
        df_mu = pd.DataFrame(data=muValues.transpose(), columns=['mu' + str(i) for i in range(modeN)])
        df = pd.concat([df_time, df_measurement, df_state, df_variance, df_error, df_mu], axis=1)

        df.to_csv(folder + simName + '_data.csv', index=False)

    fig.show()
    plt.show()


model = createFilter(stateN, measN, modeN, dt, xD, dxD, sigma_x, sigma_y, params, mu, trans, S)
test_particle_filter(model, testData, MULTI_MODEL)


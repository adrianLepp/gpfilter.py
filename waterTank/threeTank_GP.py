from filterpy.kalman import IMMEstimator
import numpy as np
import matplotlib.pyplot as plt
from threeTank import getThreeTankEquations, ThreeTank, parameter as param
from GP_BF import GP_SSM_gpy_multiout, GP_SSM_gpy_LVMOGP, GP_SSM_gpytorch_multitask, BatchIndependentMultitaskGPModel, MultitaskGPModel
from helper import init_GP_UKF, init_UKF, createTrainingData
import pandas as pd
import json
from dynamicSystem import simulateNonlinearSSM


gp_model = GP_SSM_gpytorch_multitask
gp_type =   MultitaskGPModel #  BatchIndependentMultitaskGPModel #  ConvolvedGPModel #

OPTIM_STEPS = 1
verbose = True
NORMALIZE = True

time  = 100

# -----------------------------------------------------------------------------
# create training data
#------------------------------------------------------------------------------

stateN = 3
x0 = np.zeros(stateN) # initial state
dt = 0.1

metaParams = [{'T':100, 'downsample':100}]
params = [param]

xD, yD, dxD, tsD = createTrainingData(ThreeTank, params, metaParams, stateN, dt, x0, multipleSets=False)


# -----------------------------------------------------------------------------
# create GP model
#------------------------------------------------------------------------------


model = gp_model(dxD.transpose(), yD.transpose(), stateN, normalize=NORMALIZE, model=gp_type)
model.optimize(OPTIM_STEPS, verbose)

model.observation = ThreeTank(param).observation 

model.stateTransition = model.stateTransitionDx
# -----------------------------------------------------------------------------
# simulate GP
#------------------------------------------------------------------------------

x0 = np.array([0.1, 0.01, 0.04])

xTest, yTest, dxTest, tsTest = simulateNonlinearSSM(model, x0, dt, time)


plt.figure()
for i in range(stateN):
    plt.plot(tsD, yD[i, :], label='$z_' + str(i) + '$')
    plt.plot(tsTest, xTest[i, :], '--', label='$x_' + str(i) + '$')
    #plt.fill_between(time, lower_bound[i, :], upper_bound[i, :], color='gray', alpha=0.5, label='$95\%$ CI' if i == 0 else "")
plt.legend()

plt.show()


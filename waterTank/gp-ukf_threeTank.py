#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:22:05 2024

@author: alepp
"""


from filterpy.kalman import IMMEstimator, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np
import matplotlib.pyplot as plt
from dynamicSystem import simulateNonlinearSSM
from threeTank import getThreeTankEquations, ThreeTank, parameter as param

from GP_BF import GP_SSM_gpy_multiout, init_GP_UKF, GP_SSM_gpy_LVMOGP, GP_SSM_gpytorch_multitask

stateTransition, observation = getThreeTankEquations()

# %%

# -----------------------------------------------------------------------------
# create training data
#------------------------------------------------------------------------------

stateN = 3
x0 = np.zeros(stateN) # initial state
dt = 1
# create training data through simulation
xData1, yData1, dxData1, tsData1 = simulateNonlinearSSM(ThreeTank(), x0, dt, 100)


#empty the tanks
xFull = xData1[:,xData1.shape[1]-1]

param2 = param.copy()
#param2['u'] = 0
param2['c13'] = param['c13'] * 2 
param2['c32'] = param['c32'] * 2
param2['c2R'] = param['c2R'] * 2

xData2, yData2, dxData2, tsData2 = simulateNonlinearSSM(ThreeTank(param2), xFull, dt, 500)

xData = np.concatenate((xData1, xData2), axis=1)
yData = np.concatenate((yData1, yData2), axis=1) 
dxData = np.concatenate((dxData1, dxData2), axis=1)
tsData2 += tsData1[-1]
tsData = np.concatenate((tsData1, tsData2))

plt.figure()
plt.plot(tsData, yData[0,:], label='y1')
plt.plot(tsData, yData[1,:], label='y2')
plt.plot(tsData, yData[2,:], label='y3')
plt.legend()

xData2 = xData2[:, ::5]
yData2 = yData2[:, ::5]
dxData2 = dxData2[:, ::5]
tsData2 = tsData2[::5]


yD = [yData1, yData2[:,1:-1]]
dxD = [dxData1, dxData2[:,1:-1]]
xD = [xData1, xData2[:,1:-1]]

# %%

# -----------------------------------------------------------------------------
# init Algorithm
#------------------------------------------------------------------------------

# init variables for IMM-GP-UKF


z_std = 1e-5


P = 1e-3 # initial uncertainty
R = np.diag([z_std**2, z_std**2, z_std**2]) # 1 standard
Q = Q_discrete_white_noise(dim=stateN, dt=dt, var=1e-7**2, block_size=1)

# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(stateN, alpha=.1, beta=2., kappa=-1)



# init GP-UKF filters
# GP_SSM_gpy_LVMOGP
# GP_SSM_gpytorch_multitask
model = GP_SSM_gpytorch_multitask(dxData.transpose(), yData.transpose(), stateN, normalize=False)
model.optimize(10000)
    
gp_filter  = init_GP_UKF(x0, model.stateTransition, observation, points, model.stateTransitionVariance, P, R, Q, dt)

# %%

# -----------------------------------------------------------------------------
# run prediction
#------------------------------------------------------------------------------

T = 100
x = np.zeros(stateN)

#zs = [[i+randn()*z_std, i+randn()*z_std] for i in range(T)]
xSim, ySim, dxSim, tSim = simulateNonlinearSSM(ThreeTank(param), x, dt, T)
xSim2, ySim2, dxSim2, tSim2 = simulateNonlinearSSM(ThreeTank(param2), xSim[:,-1], dt, 2*T)


T = 3*T
xSim = np.concatenate((xSim, xSim2), axis=1)
ySim = np.concatenate((ySim, ySim2), axis=1)
dxSim = np.concatenate((dxSim, dxSim2), axis=1)


#txSim = np.concatenate((tSim, tSim2), axis=1)

zValues =  ySim
xValues = np.zeros((stateN,T))

for i in range(T):
    # perform predict/update cycle
    gp_filter.predict()
    gp_filter.update(zValues[:,i])

    xValues[:,i] = gp_filter.x

plt.figure()
plt.plot(zValues[0,:],label='$z_1$')
plt.plot(zValues[1,:],label='$z_2$')
plt.plot(zValues[2,:],label='$z_3$')

plt.plot(xValues[0,:],'--',label='$x_1$')
plt.plot(xValues[1,:],'--',label='$x_2$')
plt.plot(xValues[2,:],'--',label='$x_3$')
plt.legend()
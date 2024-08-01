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

from GP_BF import GP_SSM, GP_SSM2, GP_SSM3, init_GP_UKF

# %% 
param2 = param.copy()
#param2['u'] = 0
param2['c13'] = param['c13'] * 2 
param2['c32'] = param['c32'] * 2
param2['c2R'] = param['c2R'] * 2

stateTransition1, observation = getThreeTankEquations(param)
stateTransition2, observation2 = getThreeTankEquations(param2)


stateTransition = [stateTransition1, stateTransition2]

# %%

# -----------------------------------------------------------------------------
# init Algorithm
#------------------------------------------------------------------------------

# init variables for IMM-GP-UKF
modeN = 2
stateN = 3
dt = 1
x0 = np.zeros(stateN) 

filters = []
models = []
mu = [0.9, 0.1] 
trans = np.array([[0.97, 0.03], [0.03, 0.97]])


z_std = 1e-5


P = 0.2 # initial uncertainty
R = np.diag([z_std**2, z_std**2, z_std**2]) # 1 standard
Q = Q_discrete_white_noise(dim=stateN, dt=dt, var=1e-7**2, block_size=1)

# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(stateN, alpha=.1, beta=2., kappa=-1)


def init_UKF(x, fx, hx , points, P, R, Q, dt):
    dim_x = x.shape[0]
    dim_z = R.shape[0]

    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, points=points)

    ukf.x = x
    ukf.P *= P 
    ukf.R = R 
    ukf.Q = Q

    return ukf


# init IMM with GP-UKF filters
for i in range(modeN):
    filters.append(init_UKF(x0, stateTransition[i], observation, points, P, R, Q, dt))
    

immUkf = IMMEstimator(filters, mu, trans)

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
muValues = np.zeros((modeN,T))

muV = []

for i in range(T):
    # perform predict/update cycle
    immUkf.predict()
    immUkf.update(zValues[:,i])

    xValues[:,i] = immUkf.x
    muValues[:,i] = immUkf.mu
    muV.append(immUkf.mu)
    # xValues[0,i] = immUkf.x[0]
    # xValues[1,i] = immUkf.x[1]
    # xValues[2,i] = immUkf.x[2]

plt.figure()
plt.plot(zValues[0,:],label='$z_1$')
plt.plot(zValues[1,:],label='$z_2$')
plt.plot(zValues[2,:],label='$z_3$')

plt.plot(xValues[0,:],'--',label='$x_1$')
plt.plot(xValues[1,:],'--',label='$x_2$')
plt.plot(xValues[2,:],'--',label='$x_3$')
plt.legend()

plt.figure()
plt.plot(muValues[0,:], label='$\mu^1$')
plt.plot(muValues[1,:], label='$\mu^2$')
plt.legend()
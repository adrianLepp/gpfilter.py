#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:22:05 2024

@author: alepp
"""
from filterpy.kalman import IMMEstimator
import numpy as np
import matplotlib.pyplot as plt
from dynamicSystem import simulateNonlinearSSM
from threeTank import getThreeTankEquations, ThreeTank, parameter as param
from util import createTrainingData
from GP_BF import GP_SSM_gpy_multiout, GP_SSM_gpy_LVMOGP, GP_SSM_gpytorch_multitask
from helper import init_GP_UKF, init_UKF

stateTransition, observation = getThreeTankEquations()
# %%
MULTI_MODEL = True
GP = True

NORMALIZE = True
OPTIM_STEPS = 1


gp_model = GP_SSM_gpytorch_multitask

# %%

# -----------------------------------------------------------------------------
# create training data
#------------------------------------------------------------------------------

stateN = 3
x0 = np.zeros(stateN) # initial state
dt = 1

param2 = param.copy()
#param2['u'] = 0
param2['c13'] = param['c13'] * 2 
param2['c32'] = param['c32'] * 2
param2['c2R'] = param['c2R'] * 2

metaParams = [
    {'T':100, 'downsample':1}, 
    {'T':100, 'downsample':1}
]

params = [param, param2]

if GP:
    xD, yD, dxD, tsD = createTrainingData(ThreeTank, params, metaParams, stateN, dt, x0, multipleSets=MULTI_MODEL)

# %%

# -----------------------------------------------------------------------------
# init Algorithm
#------------------------------------------------------------------------------
z_std = 1e-5
x_std = 1e-7
P = 1e-3 # initial uncertainty
modeN = 2


mu = [0.9, 0.1] 
trans = np.array([[0.97, 0.03], [0.03, 0.97]])

def createFilter(modeN:int, x_std:float, z_std:float, P:float, mu:list[float], trans:np.ndarray, observation):
# init variables for IMM-GP-UKF

    if MULTI_MODEL:
        if GP:
            filters = []
            models = []
            for i in range(modeN):
                models.append(gp_model(dxD[i].transpose(), yD[i].transpose(), stateN, normalize=NORMALIZE))
                
                # for param_name, param in models[0].gp.named_parameters():
                #     print(f'Parameter name: {param_name:42} value = {param.data}') #.item()
                
                models[i].optimize(OPTIM_STEPS)
        
                filters.append(init_GP_UKF(x0, models[i].stateTransition, observation, stateN, models[i].stateTransitionVariance,P, z_std, dt))
        
        else:
            filters = []
            for i in range(modeN):
                stateTransition, observation = getThreeTankEquations(params[i])
                filters.append(init_UKF(x0, stateTransition, observation,stateN, x_std, P, z_std, dt))

        immUkf = IMMEstimator(filters, mu, trans)
        return immUkf, models

    else:
        model = gp_model(dxD.transpose(), yD.transpose(), stateN, normalize=NORMALIZE)
        model.optimize(OPTIM_STEPS)
        gp_filter  = init_GP_UKF(x0, model.stateTransition, observation, stateN, model.stateTransitionVariance,P, z_std, dt)
        return gp_filter, model

filter, model = createFilter(modeN, x_std, z_std, P, mu , trans, observation)
    


# %%

# -----------------------------------------------------------------------------
# run prediction
#------------------------------------------------------------------------------
def runPrediction(stateN, modeN, filter, T, zValues):
    xValues = np.zeros((stateN,T))
    if MULTI_MODEL:
        muValues = np.zeros((modeN,T))

    for i in range(T):
        # perform predict/update cycle
        filter.predict()
        filter.update(zValues[:,i])

        if MULTI_MODEL:
            muValues[:,i] = filter.mu

        xValues[:,i] = filter.x

    #with plt.ion():
    plt.figure()
    for i in range(stateN):
        plt.plot(zValues[i,:],label='$z_' + str(i) + '$')
        plt.plot(xValues[i,:],'--',label='$x_' + str(i) + '$')
    plt.legend()

    if MULTI_MODEL:
        plt.figure()
        for i in range(modeN):
            plt.plot(muValues[i,:], label='$\mu^' + str(i) + '$')
        plt.legend()

        plt.show()



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

runPrediction(stateN, modeN, filter, T, zValues)

# %%
# gp_param = []
# for param_name, param in model[1].gp.named_parameters():
#     print(f'Parameter name: {param_name:42} value = {param.data}')
    
    
#     constraint = model[1].gp[param_name + '_constrained']
    #constraint.transform(param)
    #gp_param.append((param_name,param))
    #print(f'Parameter name: {param_name:42} value = {param.data}')
    



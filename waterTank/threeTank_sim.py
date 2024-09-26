#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:22:05 2024

@author: alepp
"""
from filterpy.kalman import IMMEstimator
import numpy as np
import matplotlib.pyplot as plt
from threeTank import getThreeTankEquations, ThreeTank, parameter as param
from GP_BF import GP_SSM_gpy_multiout, GP_SSM_gpy_LVMOGP, GP_SSM_gpytorch_multitask, BatchIndependentMultitaskGPModel, MultitaskGPModel
from helper import init_GP_UKF, init_UKF, createTrainingData
import pandas as pd
import json

stateTransition, observation = getThreeTankEquations(param, observe= (False, False, False))
# %%

simCounter = 11

verbose = True
MULTI_MODEL = False
GP = True
SAVE = False

NORMALIZE = True
OPTIM_STEPS = 500


gp_model = GP_SSM_gpytorch_multitask
gp_type =  MultitaskGPModel # BatchIndependentMultitaskGPModel #  

# %%

# -----------------------------------------------------------------------------
# create training data
#------------------------------------------------------------------------------

stateN = 3
measN = 1
x0 = np.zeros(stateN) # initial state
dt = 0.1

#UKF
alpha=.1
beta=2.
kappa=-1
z_std = 5*1e-7
x_std = 1e-9
P = 1e-9 # initial uncertainty

#IMMM
modeN = 2
mu = [0.9, 0.1] 
trans = np.array([[0.9999, 0.0001], [0.0001, 0.9999]])

param2 = param.copy()
param2['c13'] = param['c13'] * 4 
param2['c32'] = param['c32'] * 4
param2['c2R'] = param['c2R'] * 4
param2['u'] = 0

metaParams = [
#    {'T':100, 'downsample':100}, 
    {'T':100, 'downsample':10}
]

params = [
    param, 
#    param2
]
#params = [param]


if GP:
    xD, yD, dxD, tsD = createTrainingData(ThreeTank, params, metaParams, stateN, dt, x0, multipleSets=MULTI_MODEL)

# %%

# -----------------------------------------------------------------------------
# init Algorithm
#------------------------------------------------------------------------------

#trans = np.array([[1, 0], [0, 1]])


# data = {
#     'dx': dxD,
#     'y': yD,
# }
# filterParam = {
#     'modeN': modeN,
#     'x_std': x_std,
#     'z_std': z_std,
#     'P': P,
#     'mu': mu,
#     'trans': trans
# }

# systemEquations = {
#     'stateTransition': stateTransition,
#     'observation': observation
# }

def createFilter(modeN:int, x_std:float, z_std:float, P:float, mu:list[float], trans:np.ndarray, observation):
# init variables for IMM-GP-UKF

    if MULTI_MODEL:
        filters = []
        models = []
        if GP:
            
            for i in range(modeN):
                models.append(gp_model(dxD[i].transpose(), xD[i].transpose(), stateN, normalize=NORMALIZE, model=gp_type))
                
                # for param_name, param in models[0].gp.named_parameters():
                #     print(f'Parameter name: {param_name:42} value = {param.data}') #.item()
                
                models[i].optimize(OPTIM_STEPS, verbose)
        
                filters.append(init_GP_UKF(x0, models[i].stateTransition, observation, stateN, measN, models[i].stateTransitionVariance,P, z_std, dt))
        
        else:
            for i in range(modeN):
                stateTransition, observation = getThreeTankEquations(params[i])
                filters.append(init_UKF(x0, stateTransition, observation,stateN, x_std, P, z_std, dt, alpha, beta, kappa))

        immUkf = IMMEstimator(filters, mu, trans)
        return immUkf, models

    else:
        model = gp_model(dxD.transpose(), xD.transpose(), stateN, normalize=NORMALIZE, model=gp_type)
        model.optimize(OPTIM_STEPS, verbose)
        gp_filter  = init_GP_UKF(x0, model.stateTransition, observation, stateN, measN, model.stateTransitionVariance,P, z_std, dt)
        return gp_filter, model

filter, model = createFilter(modeN, x_std, z_std, P, mu , trans, observation)
    


# %%

# options for saving the data
settings = {
    'stateN': stateN,
    'ukf': {
        'alpha': alpha,
        'beta': beta,
        'kappa': kappa,
    },
    'z_std' : z_std,
    'x_std' : x_std,
    'P' : P,
    'dt' : dt,
}

simName = 'threeTank'
if GP:
    simName += '_gp'
    settings['gp'] ={
    'optimSteps' :OPTIM_STEPS,
    'normalize': NORMALIZE,
    'params': params,
    'metaParams': metaParams,
    } 

if MULTI_MODEL:
    simName += '_imm'
    settings['imm'] = {
        'modeN': modeN,
        'mu': mu,
        'trans': trans.tolist(),
    }

simName += '_' + str(simCounter)
folder = 'results/'

dataName = folder + simName + 'settings'

if SAVE:
    with open(folder + simName + '_settings.json',"w") as f:
        json.dump(settings,f)

#df_settings = pd.DataFrame(data=settings)
#df_settings.to_json(folder + simName + 'settings.json')

# -----------------------------------------------------------------------------
# run prediction
#------------------------------------------------------------------------------
def runPrediction(stateN, modeN, filter, zValues:np.ndarray, time:np.ndarray):
    shape = zValues.shape
    simLength = time.shape[0]#  shape[1]
    xValues = np.zeros(shape)
    errorValues = np.zeros(shape)
    likelihoodValues = np.zeros(shape)
    varianceValues = np.zeros(shape)
    if MULTI_MODEL:
        muValues = np.zeros((modeN,simLength))

    for i in range(simLength):
        # perform predict/update cycle
        filter.predict()
        meas = zValues[0,i] + zValues[1,i] + zValues[2,i] 
        filter.update(meas)#TODO change this

        if MULTI_MODEL:
            muValues[:,i] = filter.mu

        xValues[:,i] = filter.x
        errorValues[:,i] = np.abs(filter.x - zValues[:,i])
        varianceValues[:,i] = np.diag(filter.P_post)
        likelihoodValues[:,i] = filter.likelihood

    lower_bound = xValues - 1.96 * np.sqrt(varianceValues)
    upper_bound = xValues + 1.96 * np.sqrt(varianceValues)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(stateN):
        ax[0,0].plot(time, zValues[i, :], label='$z_' + str(i) + '$')
        ax[0,0].plot(time, xValues[i, :], '--', label='$x_' + str(i) + '$')
        ax[0,0].fill_between(time, lower_bound[i, :], upper_bound[i, :], color='gray', alpha=0.5, label='$95\%$ CI' if i == 0 else "")
    ax[0,0].legend()

    for i in range(stateN):
        ax[0,1].plot(time, errorValues[i, :], label='$error_' + str(i) + '$')
    ax[0,1].legend()

    ax[1,0].plot(time, likelihoodValues[0,:], label='likelihood')
    ax[1,0].legend()

    if MULTI_MODEL:
        #fig2, ax[0,1] = plt.subplots(1, 1, figsize=set_size(textWidth, 0.5,(1,1)))
        for i in range(modeN):
            ax[1,1].plot(time, muValues[i,:], label='$mu^' + str(i) + '$')
        ax[1,1].legend()
        #fig2.savefig('../gaussianProcess.tex/img/modeEstimation.pdf', format='pdf', bbox_inches='tight')

    # save data
    if SAVE:
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

param3 = param.copy()
param3['c13'] = param['c13'] * 1.5 
param3['c32'] = param['c32'] * 1.5
param3['c2R'] = param['c2R'] * 1.5

testParam = [
    param, 
    #param2
    ]

metaParams = [
    {'T':100, 'downsample':1}, 
    #{'T':500, 'downsample':1}, 
]

xSim, ySim, dxSim, tsSim = createTrainingData(ThreeTank, testParam, metaParams, stateN, dt, x0, multipleSets=False, plot=False)



#txSim = np.concatenate((tSim, tSim2), axis=1)

zValues =  ySim

runPrediction(stateN, modeN, filter, zValues, tsSim)


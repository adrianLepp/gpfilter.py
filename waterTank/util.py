#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:36:50 2024

@author: alepp


this will become a lib once
"""
import torch
import numpy as np
from dynamicSystem import simulateNonlinearSSM
import matplotlib.pyplot as plt


# %%
def createTrainingData(system, paramSets, metaParams, stateN, dt, x0, multipleSets=False):
    yD = []
    dxD = []
    xD = []
    tsD = []
    T = 0
    
    for param, metaParam in zip(paramSets, metaParams):
        if not metaParam['downsample']:
            metaParam['downsample'] = 1

        xData, yData, dxData, tsData = simulateNonlinearSSM(system(param), x0, dt, metaParam['T'])

        tsData += T
        T = tsData[-1]
        x0 = xData[:, -1]

        xD.append(xData[:, ::metaParam['downsample']])
        yD.append(yData[:, ::metaParam['downsample']])
        dxD.append(dxData[:, ::metaParam['downsample']])
        tsD.append(tsData[::metaParam['downsample']])

    xData = np.concatenate((xD), axis=1)
    yData = np.concatenate((yD), axis=1)
    dxData = np.concatenate((dxD), axis=1)
    tsData = np.concatenate((tsD))

    #with plt.ion():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    for i in range(stateN):
        ax1.plot(tsData, yData[i], label='y' + str(i))
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')
    ax1.legend()

    for i in range(stateN):
        ax2.plot(tsData, dxData[i], label='dx' + str(i))
    ax2.set_xlabel('t')
    ax2.set_ylabel('dx')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    if multipleSets:
        return xD, yD, dxD, tsD
    else:
        return xData, yData, dxData, tsData


# %% mean std

def normalize_mean_std_np(data:np.ndarray, mean:float=None, std:float=None):
    if (mean is None or std is None):
        mean = data.mean(axis=0)
        std = data.std(axis=0)

    return (data - mean) / std, (mean, std)


def denormalize_mean_std(data:np.ndarray, mean: float, std:float):
    return (data - mean) / std

# %% numpoy min max
def normalize_min_max_np(data:np.ndarray, min_val:float=None, delta:float=None):
    
    if (min_val is None or delta is None):
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        delta = max_val - min_val
        
        # check for every dimension if diff=0 and replace with 1
        try:
            for i in range(delta.shape[0]):
                if delta[i] == 0:
                    delta[i] = 1
        except:
            if delta == 0:
                delta = 1

    normalized_data = (data - min_val) / delta #(max_val - min_val)
    return normalized_data, (min_val, delta)

def denormalize_min_max(data, min_val:float, delta:float):
    return data * delta + min_val

# %% pytorch min max

def normalize_min_max_torch(data:torch.tensor, min_val:float=None, delta:float=None):
    
    if (min_val is None or delta is None):
        min_val = data.min(dim=0)[0]
        max_val = data.max(dim=0)[0]
        delta = max_val - min_val
        
        # check for every dimension if diff=0 and replace with 1
        try:
            for i in range(delta.shape[0]):
                if delta[i] == 0:
                    delta[i] = 1
        except:
            if delta == 0:
                delta = 1

    normalized_data = (data - min_val) / delta #(max_val - min_val)
    return normalized_data, (min_val, delta)


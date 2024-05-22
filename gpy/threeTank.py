import numpy as np
from numpy import sqrt, sign, random

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

class ThreeTank():

    def __init__(self, param):
        self.u = param['u']
        self.c13 = param['c13']
        self.c32 = param['c32']
        self.c2R = param['c2R']
        self.A = param['A']
        self.g = param['g']

        self.sigmaX = param['sigmaX']
        self.sigmaY = param['sigmaY']

    def stateTransition(self,t, x):
        for i in range(0, len(x)):
            if x[i] < 0:
                x[i] = 0

        dx1 = 1/self.A*(self.u-self.c13*sign(x[0]-x[2])*sqrt(2*self.g*abs(x[0]-x[2])))
        dx2 = 1/self.A*(self.c32*sign(x[2]-x[1])*sqrt(2*self.g*abs(x[2]-x[1]))-self.c2R*sqrt(2*self.g*abs(x[1])))
        dx3 = 1/self.A*(self.c13*sign(x[0]-x[2])*sqrt(2*self.g*abs(x[0]-x[2]))-self.c32*sign(x[2]-x[1])*sqrt(2*self.g*abs(x[2]-x[1])))

        dx1_noisy = dx1 + random.normal(0, 1) * self.sigmaX
        dx2_noisy = dx2 + random.normal(0, 1) * self.sigmaX
        dx3_noisy = dx3 + random.normal(0, 1) * self.sigmaX
        
        return [dx1_noisy, dx2_noisy, dx3_noisy]
    
    def observation(self, x):
        y1 = x[0] + random.normal(0, 1) * self.sigmaY
        y2 = x[1] + random.normal(0, 1) * self.sigmaY
        y3 = x[2] + random.normal(0, 1) * self.sigmaY
        return [y1 , y2, y3]


def stateTransition(x, dt):
    for i in range(0, len(x)):
        if x[i] < 0:
            x[i] = 0

    dx1 = 1/parameter['A']*(parameter['u']-parameter['c13']*sign(x[0]-x[2])*sqrt(2*parameter['g']*abs(x[0]-x[2])))
    dx2 = 1/parameter['A']*(parameter['c32']*sign(x[2]-x[1])*sqrt(2*parameter['g']*abs(x[2]-x[1]))-parameter['c2R']*sqrt(2*parameter['g']*abs(x[1])))
    dx3 = 1/parameter['A']*(parameter['c13']*sign(x[0]-x[2])*sqrt(2*parameter['g']*abs(x[0]-x[2]))-parameter['c32']*sign(x[2]-x[1])*sqrt(2*parameter['g']*abs(x[2]-x[1])))

    dx1_noisy = dx1 + random.normal(0, 1) * parameter['sigmaX']
    dx2_noisy = dx2 + random.normal(0, 1) * parameter['sigmaX']
    dx3_noisy = dx3 + random.normal(0, 1) * parameter['sigmaX']
    
    return [
        dx1_noisy * dt + x[0], 
        dx2_noisy * dt + x[1], 
        dx3_noisy * dt + x[2]
    ]
    
def observation(x):
    y1 = x[0] + random.normal(0, 1) * parameter['sigmaY']
    y2 = x[1] + random.normal(0, 1) * parameter['sigmaY']
    y3 = x[2] + random.normal(0, 1) * parameter['sigmaY']
    return [y1 , y2, y3]
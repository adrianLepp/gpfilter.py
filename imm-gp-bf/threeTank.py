import numpy as np
from numpy import sqrt, sign, random
from types import SimpleNamespace

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

param = SimpleNamespace(**parameter)

'''
nonlinear system equations for Three Tank to simulate with scipy.integrate.solve_ivp
stateTransition returns the derivative of the state vector x and has the extra argument to for time invariant systems
'''
class ThreeTank():

    def __init__(self, param=parameter):
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
    
    def observation(self, x, observe = (True, True, True)):
        y = []
        for i in range(0, len(x)):
            if observe[i]:
                y.append(x[i] + random.normal(0, 1) * self.sigmaY)
        if len(y) == 0:
            return [x[0] + x[1] + x[2] + random.normal(0, 1) * self.sigmaY ]
        return y
    
    
def getThreeTankEquations(param=parameter, observe= (True, True, True)):
    threeTank  = ThreeTank(param)

    '''
    nonlinear state transition for the three tank system
    returns the next state vector for the given state vector x and time step dt
    this is used by filterPy
    '''
    def stateTransition(x, dt):
        dx = threeTank.stateTransition(0,x)
        return np.add(x, np.multiply(dx, dt))

    '''
    nonlinear state transition for the three tank system
    returns the observation for the given state vector x
    this is used by filterPy
    '''
    def observation(x):
        return threeTank.observation(x, observe)
    
    return stateTransition, observation

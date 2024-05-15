from scipy.integrate import solve_ivp
from numpy import sqrt, sign, abs, random
import numpy as np

def simulateNonlinearSSM(system, x0, dt, tFinal):
    t0 = 0
    ts = np.arange(t0, tFinal, dt)

    sol = solve_ivp(system.stateTransition, [t0, tFinal], x0, method='RK45', t_eval=ts, max_step=dt, atol = 1, rtol = 1)

    x = sol.y

    y = np.zeros((3,len(ts)))
    dx = np.zeros((3,len(ts)))
    xPrev = x0
    for i in range(len(ts)):
        y[:,i] = system.observation(x[:,i]) 
        dx[:,i] = (x[:,i]-xPrev)
        xPrev = x[:,i]

    return x, y, dx, ts

def create_function(param):
    def inner_function(arg1, arg2):
        # Do something with param, arg1, and arg2
        # ...
        return 1
    
    return inner_function
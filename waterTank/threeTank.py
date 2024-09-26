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
        # y1 = x[0] + random.normal(0, 1) * self.sigmaY
        # y2 = x[1] + random.normal(0, 1) * self.sigmaY
        # y3 = x[2] + random.normal(0, 1) * self.sigmaY
        # return [y1 , y2, y3]

        return y
    
    
def getThreeTankEquations(param=parameter, observe= (True, True, True)):
    threeTank  = ThreeTank(param)

    '''
    nonlinear state transition for the three tank system
    returns the next state vector for the given state vector x and time step dt
    this is used by filterPy
    TODO: One should give the option for different solvers like RK45 (this is one step euler method)
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



# linearized form
# calculations are from Hennin Borchard, Praktikum Zustandsregelungen 2020, Hochschule Bielefeld

x = [0,0,0]

# rest position
u_r  = parameter['u']*0.3 



x_r1=(1+2*(parameter['c2R']/parameter['c32'])**2)*(1)/(2*parameter['g']*parameter['c2R']**2)*u_r**2
x_r2=u_r**2/(2*parameter['g']*parameter['c2R']**2)
x_r3=(1+(parameter['c2R']/parameter['c32'])**2)*(1)/(2*parameter['g']*parameter['c2R']**2)*u_r**2

x_r=[x_r1, x_r2, x_r3]

# linear system matrix in rest position:

# dx  =Ax + Bu
# dx = A_r * (x - x_r) + B_r * (u - u_r)

# first line
a11=-parameter['c13']*parameter['g']/parameter['A']*1/(sqrt(2*parameter['g']*(x_r1-x_r3)))
a12=0
a13=parameter['c13']*parameter['g']/parameter['A']*1/(sqrt(2*parameter['g']*(x_r1-x_r3)))

#second line
a21=0
a22=-parameter['c32']*parameter['g']/parameter['A']*1/(sqrt(2*parameter['g']*(x_r3-x_r2)))-(parameter['c2R']*parameter['g'])/parameter['A']*1/(sqrt(2*parameter['g']*x_r2))
a23=parameter['c32']*parameter['g']/parameter['A']*1/(sqrt(2*parameter['g']*(x_r3-x_r2)))

#third line
a31=parameter['c13']*parameter['g']/parameter['A']*1/(sqrt(2*parameter['g']*(x_r1-x_r3)))
a32=parameter['c32']*parameter['g']/parameter['A']*1/(sqrt(2*parameter['g']*(x_r3-x_r2)))
a33=-parameter['c13']*parameter['g']/parameter['A']*1/(sqrt(2*parameter['g']*(x_r1-x_r3)))-parameter['c32']*parameter['g']/parameter['A']*1/(sqrt(2*parameter['g']*(x_r3-x_r2))) 

A_r=[[a11, a12, a13],
   [a12, a22 ,a23],
   [a31, a32 ,a33]]

b_r=[1/parameter['A'], 0, 0]   #df/du

# x_d = x - x_r
# u_d = parameter['u'] - u_r

# change the form of the system to A_t = [x; u] = 0, where A_t contains differential operators
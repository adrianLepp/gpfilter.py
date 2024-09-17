
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np
from GP_BF import GP_UKF
from dynamicSystem import simulateNonlinearSSM
import matplotlib.pyplot as plt
from plotConfig import set_size

#plt.style.use('seaborn')
#plt.style.use('tex')
# in LaTex show the textWidth with '\the\textwidth'
textWidth= 469.4704

#--------------------------------------------------------------------------------
# init Filters
#------------------------------------------------------------------------------

def init_GP_UKF(x, fx, hx, n , Qfct, P, z_std, dt, alpha=.1, beta=2., kappa=-1):
    dim_x = x.shape[0]
    dim_z = x.shape[0]
    R = np.diag([z_std**2] * n)

    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(n, alpha, beta, kappa)

    gp_ukf = GP_UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, Qfct=Qfct, points=points)
    #              sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
    #              residual_x=None,
    #              residual_z=None,
    #              state_add=None)

    gp_ukf.x = x
    gp_ukf.P *= P
    gp_ukf.R = R 
    gp_ukf.Q = Qfct(x)

    return gp_ukf

def init_UKF(x, fx, hx , n, x_std, P, z_std, dt, alpha=.1, beta=2., kappa=-1):
    dim_x = x.shape[0]
    dim_z = x.shape[0]
    R = np.diag([z_std**2] * n)
    points = MerweScaledSigmaPoints(n, alpha, beta, kappa)

    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, points=points)

    ukf.x = x
    ukf.P *= P 
    ukf.R = R 
    ukf.Q = Q_discrete_white_noise(dim=n, dt=dt, var=x_std**2, block_size=1)

    return ukf
    

#--------------------------------------------------------------------------------
# create training data
#------------------------------------------------------------------------------

# %%
def createTrainingData(system, paramSets, metaParams, stateN, dt, x0, multipleSets=False, plot=True):
    yD = []
    dxD = []
    xD = []
    tsD = []
    T = 0
    
    for param, metaParam in zip(paramSets, metaParams):
        if not metaParam['downsample']:
            metaParam['downsample'] = 1

        xData, yData, dxData, tsData = simulateNonlinearSSM(system(param), x0, dt, metaParam['T'])

        dxData[:, 0] = xData[:, 0] - x0
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
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(textWidth, 0.5,(1,2)))

        for i in range(stateN):
            ax1.plot(tsData, yData[i], 'x', label='y' + str(i))
        ax1.set_xlabel('t')
        ax1.set_ylabel('y')
        ax1.legend()

        for i in range(stateN):
            ax2.plot(tsData, dxData[i], 'x', label='dx' + str(i))
        ax2.set_xlabel('t')
        ax2.set_ylabel('dx')
        ax2.legend()

        #fig.tight_layout()
        fig.show()
        fig.suptitle('Training Data')

        #fig.savefig('../gaussianProcess.tex/img/TrainingData.pdf', format='pdf', bbox_inches='tight')#FIXME

    if multipleSets:
        return xD, yD, dxD, tsD
    else:
        return xData, yData, dxData, tsData
import numpy as np
import scipy.special as ss

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def curve(tt, P):
    '''
    # generate an n-th order Bezier curve (order depends on number of control points)
    # the number of samples depends on length of tt

    tt: array; independent variable, tt = [0,...,1]
    P: ndarray; xyz of the control points, P = [[x0,y0,z0],[x1,y1,z1], ..., [xn,yn,zn]]
    '''

    n = P.shape[0]-1
    B = 0
    for k in range(0,n+1):
        B = B + np.squeeze([ss.comb(n,k)*(t**k)*(1-t)**(n-k)*P[k,:] for t in tt])
    
    return B

def length(B):
    '''
    # return the length of a curve given in xyz sample points.

    P: ndarray; xyz sample points of a curve, B = [[x0,y0,z0],[x1,y1,z1], ..., [xm,ym,zm]]
    '''

    L = np.sum([np.sqrt(b@b) for b in np.diff(B, axis=0)])    
    return L


if __name__ == '__main__':
    P = np.array([[0,0,0],[1,0,1],[2,0,0]])
    tt = np.linspace(0,1,100)
    B = curve(tt, P)
    l = length(B)
    fig = plt.figure()                          # initiat a figure
    ax = fig.add_subplot(111, projection='3d')  # create a 3D axes in the figure
    
    ax.plot(B[:,0], B[:,1], B[:,2])
    
    P = np.array([[0,0.1,0],[1,0.1,1],[2,0.1,0]])
    tt = np.linspace(0,1,100)
    B = curve(tt, P)
    ax.plot(B[:,0], B[:,1], B[:,2])
    
    P = np.array([[0,-0.1,0],[1,-0.1,1],[2,-0.1,0]])
    tt = np.linspace(0,1,100)
    B = curve(tt, P)
    ax.plot(B[:,0], B[:,1], B[:,2])
    
    ax.set_xlim3d(0,2)
    ax.set_ylim3d(-0.2,0.2)
    ax.set_zlim3d(0,2)
    plt.show()
    
# EOF
#Implementation of Random features and comparision with kernel methods
import math
import numpy as np
from scipy.special import erfinv
import sys
if '.' not in sys.path:
    sys.path.append('../')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhs import RKHS
from Probability.Prob import ProbSpace
from Condtest2 import *
from rffgpr import *

testMin = 0
testMax = 2

def get_rffs(X,rff_dim,sigma=0.2):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """
        np.random.seed(2)
        N, D = X.shape
        #print(X.shape)
        
        W = np.random.normal(loc=0, scale=1, size=(rff_dim, D))
        b = np.random.uniform(0, 2*np.pi, size=rff_dim)

        B    = np.repeat(b[:, np.newaxis], N, axis=1)
        norm = 1./ np.sqrt(rff_dim)
        Z    = norm * np.sqrt(2) * np.cos(sigma * W @ X.T + B)
        return Z, W, b

def m(x,r1,r2):
    sum1 = 0    
    sum2 = 0
    size = len(r1.X)
    for i in range(size):        
        sum1 += r1.K(x,r1.X[i]) 
        sum2 += r1.K(x,r1.X[i]) * r2.X[i]
    
    return sum2/sum1

def rff(x,w,X,Y,datasize):
    R,_ = shape(w)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(datasize):
        for j in range(R):
            sum1 += math.cos(w[j]*(x-X[i]))
        sum1 = sum1/R
        sum2 += sum1
        sum3 += sum1*Y[i]
        sum1 =0
    return sum3/sum2 
    



if __name__ == '__main__':
    path = '../models/Cprobdata.csv'
    d = getData.DataReader(path)
    data = d.read()        
    X1 = data['X']
    Y1 = data['Y']

    Datasize = 10000
    Featsize = 100
    Featsize2 = int(Datasize/10)

    X = np.reshape(X1[:Datasize],(Datasize,1))
    Y = np.reshape(Y1[:Datasize],(Datasize,1))

    s = 0.2 #sigma value for RKHS

    r1 = RKHS(X1,kparms=[s])
    r2 = RKHS(Y1,kparms=[s])


    R = RFFGaussianProcessRegressor(rff_dim=Featsize,sigma=0.2)
    #R.fit(X,Y)

    Z,W,b = get_rffs(X,Featsize)
    # print(shape(W))
    # print("x=-1, y=",rff(2,W,X,Y,Datasize))



    testPoints = []
    tp = testMin
    numTP = 20
    interval = (testMax - testMin) / numTP
    sq = []
    rf = []
    Err = []

    for i in range(numTP + 1):
        testPoints.append(tp)                
        # p = ps.distr('Y', [('X', tp)]).E()        
        # probpy.append(p)
        print(i)
        ans = rff(tp,W,X,Y,Datasize)
        rf.append(ans)        
        r = square(tp)
        sq.append(r)
        err1 = abs(r-ans)
        Err.append(err1)        
             
        #print(tp,p,r,err1)
        tp += interval

    plt.plot(testPoints,sq, label = 'Ideal Curve')
    plt.plot(testPoints,rf, label = 'RFF')    
    plt.legend()
    plt.show()

    


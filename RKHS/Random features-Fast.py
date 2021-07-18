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

testMin = -10
testMax = 10

if __name__ == '__main__':
    path = '../models/Cprobdata.csv'
    d = getData.DataReader(path)
    data = d.read()        
    X1 = data['X']
    Y1 = data['Y']

    Datasize = 10000
    Featsize = 10
    Featsize2 = int(Datasize/10)

    X = np.reshape(X1[:Datasize],(Datasize,1))
    Y = np.reshape(Y1[:Datasize],(Datasize,1))
    Trkhs = 0
    Trff = 0
    Tprob = 0
    

    s = 0.2 #sigma value for RKHS
    
    t1 = time.time()
    r1 = RKHS(X1,kparms=[s])
    r2 = RKHS(Y1,kparms=[s])
    t2 = time.time()
    Trkhs += (t2-t1)

    t1 = time.time()
    P = ProbSpace(data)
    t2 = time.time()
    Tprob += (t2-t1)

    testPoints = []
    tp = testMin
    numTP = 200
    interval = (testMax - testMin) / numTP
    sq = []
    Probpy = []
    Err = []
    cond = []

    for i in range(numTP + 1):
        testPoints.append(tp)                
        
        t1 = time.time()
        ans = m(tp,r1,r2)
        if ans != 0:
            cond.append(ans)
        elif i== 0:
            cond.append(0)
        else:
            cond.append(cond[i-1])
        t2 = time.time()
        Trkhs =Trkhs + (t2-t1)
        
        t1 = time.time()        
        p = P.distr('Y', [('X', tp)]).E()        
        Probpy.append(p)
        t2 = time.time()
        Tprob = Tprob + (t2-t1)       
        r = square(tp)
        sq.append(r)        
        tp += interval


    R = RFFGaussianProcessRegressor(rff_dim=Featsize,sigma=0.2)
    R.fit(X,Y)
    t1 = time.time()
    TPS = np.reshape(testPoints[:numTP+1],(numTP+1,1))
    matrff,garbage = R.predict(TPS)
    t2= time.time()
    Trff = t2-t1

    print("RFF time:\t",Trff)
    print("RKHS time:\t",Trkhs)
    print("Probpy time:\t",Tprob)

    plt.plot(testPoints,sq, label = 'Ideal Curve')
    plt.plot(testPoints,cond, label = 'RKHS Curve')
    plt.plot(testPoints,Probpy, label = 'Proobpy Curve')
    plt.plot(testPoints,matrff, label = 'RFF')    
    plt.legend()
    plt.show()

    


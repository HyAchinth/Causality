#Implementation of Random features and comparision with kernel methods
#from RKHS.filterRKHS import RSqerr, Sqmean
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
    Datasize = 1000
    
    path = '../models/Cprobdata.csv'
    d = getData.DataReader(path)
    data = d.read()[:Datasize]        
    X1 = data['X']
    Y1 = data['Y']

    Featsize = 100
    

    X = np.reshape(X1[:Datasize],(Datasize,1))
    Y = np.reshape(Y1[:Datasize],(Datasize,1))
    Trkhs = 0
    Trff = 0
    Tprob = 0
    

    s = 0.2 #sigma value for RKHS
    
    t1 = time.time()
    r1 = RKHS(X1[:Datasize],kparms=[s])
    r2 = RKHS(Y1[:Datasize],kparms=[s])
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

    #RFF calculation
    t1 = time.time()
    R = RFFGaussianProcessRegressor(rff_dim=Featsize,sigma=0.2)
    R.fit(X,Y)
    t2 = time.time()
    print("RFF overhead:",t2-t1)
    t1 = time.time()
    TPS = np.reshape(testPoints[:numTP+1],(numTP+1,1))
    matrff,garbage = R.predict(TPS)
    t2= time.time()
    Trff = t2-t1

    #Errors Calculation

    Rdev= []
    Pdev = []
    Fdev = []

    
    for i in range(numTP + 1):
        ans = sq[i]
        r = cond[i]
        p = Probpy[i]
        ff = matrff[i]
        err = abs(r-ans)
        Rdev.append(err)
        err2 = abs(p-ans)
        Pdev.append(err2)
        err3 = abs(ff - ans)
        Fdev.append(err3)

    ymean = sum(sq)/len(sq)

    RSqerr = []
    PSqerr = []
    FSqerr = []
    Sqmean = []

    for i in range(numTP + 1):
        RSqerr.append(Rdev[i]**2)
        PSqerr.append(Pdev[i]**2)
        FSqerr.append(Fdev[i]**2)
        Sqmean.append((sq[i]-ymean)**2)

    R2rkhs = 1 - sum(RSqerr)/sum(Sqmean)
    R2probpy = 1 - sum(PSqerr)/sum(Sqmean)
    R2rff = 1 - sum(FSqerr)/sum(Sqmean)


    Aerr = sum(Fdev)/len(Fdev)

    print("RFF time:\t",Trff,"\tR2 = ",R2rff[0],"\tavg err:",Aerr[0])
    print("RKHS time:\t",Trkhs,"\tR2 = ",R2rkhs,"\tavg err:",sum(Rdev)/len(Rdev))
    print("Probpy time:\t",Tprob,"\tR2 = ",R2probpy,"\tavg err:",sum(Pdev)/len(Pdev))

    rffstr = 'RFF F=' + str(Featsize)
    plt.plot(testPoints,sq, label = 'Ideal Curve')
    plt.plot(testPoints,cond, label = 'RKHS Curve')
    plt.plot(testPoints,Probpy, label = 'Proobpy Curve')
    plt.plot(testPoints,matrff, label = rffstr )    
    plt.legend()
    plt.show()

    


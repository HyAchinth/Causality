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

def K(x1,x2=0,kparms=[]):
    sigma = kparms[0]
    return ((1/(sigma * math.sqrt(2*math.pi))) * math.exp( -1 * (x1-x2)**2 / (2 * sigma**2) ))

def F(x,X,kparms=[]):    
    sum = 0.0
    for i in X:
        val = F(i,x,kparms)
        sum += val
    return sum/len(X)

# Ideal curve for the dataset ( Y = X^2)
def square(x):
    return x**2 

# if __name__ == '__main__':
#     path = '../models/Cprobdata.csv'
#     d = getData.DataReader(path)
#     data = d.read()        
#     X = data['X']
#     Y = data['Y']

#     ps = ProbSpace(data)
    
#     r = RKHS(Y,k = K, f = F, kparms=[2])

    # testPoints = []
    # testMin = -5
    # testMax = 5
    # tp = testMin
    # numTP = 200
    # interval = (testMax - testMin) / numTP
    # sq = []
    # probpy = []
    # cond = []
    
#     # Generate a uniform range of test points, expected curve, prob.py curve
    # for i in range(numTP + 1):
    #     testPoints.append(tp)
    #     y = square(tp)
    #     sq.append(y)
    #     p = ps.distr('Y', [('X', tp)]).E()
    #     probpy.append(p)        
    #     tp += interval

#     #Generate the conditional distribution using RKHS
#     # for i in range(len(testPoints)):
#     #     p = testPoints[i]
#     #     calcB(testPoints[i])            
#     #     fp = r.F(p)
#     #     cond.append(fp)


#     #plot graph
    # plt.plot(testPoints,sq, label = 'Ideal')
    # plt.plot(testPoints,probpy, label = 'ProbSpace')
    # plt.legend()
    # plt.show()
    
def evalyx(y,x,r1,r2): #function to calculated E(y|x)
    #sum1 contains sum(f(x)) and sum2 contains sum(f(y,x))  
    sum1 = 0    
    sum2 = 0
    size = len(r1.X)
    for i in range(size):
        t1 = r1.K(x,r1.X[i])
        sum1 += t1

        t2 = r2.K(y,r2.X[i])        
        sum2 += t1*t2
        #print("x=",x,"y=",y,"X=",r1.X[i],"Y=",r2.X[i])
    
    # print("Sum-P(y,x) =",sum2)
    # print("Sum-P(x) =",sum1)
    return sum2/sum1
        

if __name__ == '__main__':
    path = '../models/Cprobdata.csv'
    d = getData.DataReader(path)
    data = d.read()        
    X = data['X']
    Y = data['Y']

    ps = ProbSpace(data)

    size = len(X)
    sigma = 1 / math.log(size, 4)

    r1 = RKHS(X,kparms=[sigma])
    r2 = RKHS(Y,kparms=[sigma])

    x = 0.7
    y = 0.04

    print("P(y=",y,"|x=",x,") =",evalyx(y,x,r1,r2))
        
    testPoints = []
    testMin = -2
    testMax = 2
    tp = testMin
    numTP = 200
    interval = (testMax - testMin) / numTP
    sq = []
    probpy = []
    cond = []

    for i in range(numTP + 1):
        testPoints.append(tp)
        
        p = ps.P(('Y', tp), [('X', x)])
        probpy.append(p)
        e = evalyx(tp,x,r1,r2)
        cond.append(e)       
        tp += interval
    
    plt.plot(testPoints,cond, label = 'RKHS y|x='+str(x))
    plt.plot(testPoints,probpy, label = 'ProbSpace y|x='+str(x))    
    plt.legend()
    plt.show()


    print("P(y=",y,"|x=",x,") =",evalyx(y,x,r1,r2))
    
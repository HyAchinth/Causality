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

B = []

def calcB(x):
    global B
    B = 0
    return 

def E(y,x):
    calcB(x)
    #result = F(y)
    
    #return

def K(x1,x2=0,kparms=[]):
    sigma = kparms[0]
    return ((1/(sigma * math.sqrt(2*math.pi))) * math.exp( -1 * (x1-x2)**2 / (2 * sigma**2) ))

def F(x,X,kparms=[]):    
    sum = 0.0
    for i in X:
        val = F(i,x,kparms)
        sum += val*B[i]
    return sum

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

#     testPoints = []
#     testMin = -5
#     testMax = 5
#     tp = testMin
#     numTP = 200
#     interval = (testMax - testMin) / numTP
#     sq = []
#     probpy = []
#     cond = []
    
#     # Generate a uniform range of test points, expected curve, prob.py curve
#     for i in range(numTP + 1):
#         testPoints.append(tp)
#         y = square(tp)
#         sq.append(y)
#         p = ps.distr('Y', [('X', tp)]).E()
#         probpy.append(p)        
#         tp += interval

#     #Generate the conditional distribution using RKHS
#     # for i in range(len(testPoints)):
#     #     p = testPoints[i]
#     #     calcB(testPoints[i])            
#     #     fp = r.F(p)
#     #     cond.append(fp)


#     #plot graph
#     plt.plot(testPoints,sq, label = 'Ideal')
#     plt.plot(testPoints,probpy, label = 'ProbSpace')
#     plt.legend()
#     plt.show()
    
def evalyx(y,x,r1,r2):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    size = len(r1.X)
    for i in range(size):
        t1 = r1.K(x,r1.X[i])
        sum1 += t1
        t2 = r2.K(y,r2.X[i])
        sum2 += t2
        sum3 += t1*t2
        #print("t1=",t1,"t2=",t2)
    
    print("Sum-P(y,x) =",sum3)
    print("Sum-P(x) =",sum1)
    return sum3/sum1
        

if __name__ == '__main__':
    path = '../models/Cprobdata.csv'
    d = getData.DataReader(path)
    data = d.read()        
    X = data['X']
    Y = data['Y']

    size = len(X)
    sigma = 1 / math.log(size, 4)

    r1 = RKHS(X,kparms=[sigma])
    r2 = RKHS(Y,kparms=[sigma])

    x = 0.5
    y = 0.1
    
    print("P(y=",y,"|x=",x,") =",evalyx(4,2,r1,r2))
    
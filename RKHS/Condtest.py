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
        

def evalYx(x,r1,r2):
    musum = psum = 0.0
    for y in np.arange(-30,30,0.1):
        #print(y)
        PY_X = evalyx(y,x,r1,r2)
        musum += PY_X * y
        psum += PY_X
    return musum/psum



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

    x = 1
    y = 0.04

    #print("P(y=",y,"|x=",x,") =",evalyx(y,x,r1,r2))
        
    testPoints = []
    testMin = -5
    testMax = 5
    tp = testMin
    numTP = 200
    interval = (testMax - testMin) / numTP
    sq = []
    probpy = []
    cond = []
    start = time.time()
    Perror = 0.0
    Rerror = 0.0
    Pdev = []
    Rdev = []
    for i in range(numTP + 1):
        testPoints.append(tp)        
        #p = ps.P(('Y', tp), [('X', x)])
        p = ps.distr('Y', [('X', tp)]).E()        
        probpy.append(p)
        e = evalYx(tp,r1,r2)        
        cond.append(e)
        r = square(tp)
        sq.append(r)
        err1 = abs(r-p)
        Pdev.append(err1)
        err2 = abs(r-e)
        Rdev.append(err2)
        Perror += err1
        Rerror += err2
        print(tp,e,p,r,err1,err2)
        tp += interval
    
    Pavg = Perror/numTP
    Ravg = Rerror/numTP

    print("Prob.py Average Error:",Pavg,sum(Pdev)/numTP,"Max error:",max(Pdev))
    print("RKHS Average Error:",Ravg,sum(Rdev)/numTP,"Max error:",max(Rdev))

    plt.plot(testPoints,cond, label = 'RKHS Y|X')
    plt.plot(testPoints,sq, label = 'Ideal Curve')
    plt.plot(testPoints,probpy, label = 'ProbSpace Y|X')    
    plt.legend()
    plt.show()
    end = time.time()
    print('elapsed = ', end - start)


    #print("P(y=",y,"|x=",x,") =",evalYx(x,r1,r2))
    
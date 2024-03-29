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

  
def evalyx(y,x,r1,r2): #function to calculated E(y|x)
    #sum1 contains sum(f(x)) and sum2 contains sum(f(y,x))  
    sum1 = 0    
    sum2 = 0
    size = len(r1.X)
    r1bound = r1.kparms[0] * 3
    r2bound = r2.kparms[0] * 3
    for i in range(size):
        if(abs(r1.X[i] - x) <= r1bound and abs(r2.X[i]-y) <= r2bound):
            t1 = r1.K(x,r1.X[i])
            sum1 += t1        
            t2 = r2.K(y,r2.X[i])        
            sum2 += t1*t2      
    if sum1 == 0:
        return 0
    return sum2/sum1
        

def evalYx(x,r1,r2,min=1,max=10,choice=1,e=[]):
    #calculate E(Y|x)
    musum = psum = 0.0
    step = 1
    for y in np.arange(min**2-min,max**2+max,step):        
        if choice == 1:
            # print("run1")
            PY_X = evalyx(y,x,r1,r2)
        else:
            # print("run2")
            PY_X = f(y,x,r1,r2,e)
        musum += PY_X * y
        psum += PY_X
    if psum == 0: 
        return 0
    return musum/psum


#Functions for the 2step Estimator method
#Final result is f (y | x)= g(y − m(x) | x)
#so we define g and m functions:

def m(x,r1,r2):
    sum1 = 0    
    sum2 = 0
    size = len(r1.X)
    for i in range(size):        
        sum1 += r1.K(x,r1.X[i]) 
        sum2 += r1.K(x,r1.X[i]) * r2.X[i]
    if sum1 ==0:
        return 0
    return sum2/sum1

def residual(r1,r2):
    e = []
    size = len(r2.X)
    for i in range(size):
        
        e.append(r2.X[i]-m(r1.X[i],r1,r2))
    return e

def f(y,x,r1,r2,e):    
    sum1 = 0    
    sum2 = 0
    size = len(r1.X)
    r1bound = r1.kparms[0] * 3
    r2bound = r2.kparms[0] * 3
    for i in range(size):
        if(abs(r1.X[i] - x) <= r1bound and abs(r2.X[i]-y) <= r2bound):
            t1 = r1.K(x,r1.X[i])
            sum1 += t1
            t2 = r2.K(y - m(x,r1,r2), e[i])
            sum2 += t1*t2
    if sum1 ==0:
        return 0
    return sum2/sum1



if __name__ == '__main__':
    path = '../models/Cprobdata.csv'
    d = getData.DataReader(path)
    data = d.read()        
    X1 = data['X']
    Y1 = data['Y']
    X = X1[:7500]
    Y = Y1[:7500]
    ps = ProbSpace(data)

    size = len(X)    
   
    testPoints = []
    testMin = 5
    testMax = 7
    tp = testMin
    numTP = 100
    interval = (testMax - testMin) / numTP
    sq = []
    probpy = []
    cond = []
    
    Perror = 0.0    
    Pdev = []

    start1 = time.time()

    for i in range(numTP + 1):
        testPoints.append(tp)                
        p = ps.distr('Y', [('X', tp)]).E()        
        probpy.append(p)        
        r = square(tp)
        sq.append(r)
        err1 = abs(r-p)
        Pdev.append(err1)        
        Perror += err1        
        #print(tp,p,r,err1)
        tp += interval

    end1 = time.time()
    print("Prob.py Time:", end1-start1)

    Pavg = Perror/numTP

    sigma = [0.2]

    SigmaErrors = []
    MaxErrors = []
    traces = []
    times = []

    choice = [0,1]
    j=0
    e = []

    file = 'residual.txt'


    s = 1* size**(-1/5)
    R1 = RKHS(X1,kparms=[s])
    R2 = RKHS(Y1,kparms=[s])

    readfile = open("residual.txt","r")
    e = readfile.read().splitlines()
    for i in range(len(e)):
        e[i] = float(e[i])
    readfile.close()
    
    if len(e) == 0:
        t1 = time.time()
        e = residual(R1,R2)
        filename = open("residual.txt","w")    
        np.savetxt(filename,e)
        filename.close()        
        t2 = time.time()
        print("Residual Calculation time: ",str(t2-t1))    

    for s in range(len(sigma)):
        r1 = RKHS(X1,kparms=[sigma[s]])
        r2 = RKHS(Y1,kparms=[sigma[s]])
        
        # t1 = time.time()
        # new=f(4,2,r1,r2,e)
        # t2 = time.time()
        # print("2 Step estimator time: "+str(t2-t1),str(new))
        # t1 = time.time()
        # old=evalyx(4,2,r1,r2)
        # t2 = time.time()
        # print("1 step estimator time: "+str(t2-t1),str(old))


        testPoints = []
        testMin = 5
        testMax = 7
        tp = testMin
        numTP = 100
        interval = (testMax - testMin) / numTP
        
        Rdev = []
        Rdev2 = []
        cond = []
        cond2 = []
        onestep = 0
        twostep = 0

        for i in range(numTP + 1):
            testPoints.append(tp)   
            start = time.time()
            ans = evalYx(tp,r1,r2,choice=0,e=e)
            #ans = m(tp,r1,r2)
            if ans != 0:
                cond.append(ans)
            elif i== 0:
                cond.append(0)
            else:
                cond.append(cond[i-1])
            end = time.time()            
            twostep += (end-start)
            start2 = time.time()
            ans2 = evalYx(tp,r1,r2,choice=1,e=e)
            if ans != 0:
                cond2.append(ans2)
            elif i== 0:
                cond2.append(0)
            else:
                cond2.append(cond2[i-1])
            end2 = time.time()
            onestep += (end2-start2)
            r = square(tp)            
            err = abs(r-ans)
            err2 = abs(r-ans2)
            Rdev.append(err)
            Rdev2.append(err2)           
            print(tp,ans,r,err)
            print(tp,ans2,r,err2)
            tp += interval

        
        times.append(end - start)
        
        Ravg = sum(Rdev)/numTP
        Ravg2 = sum(Rdev2)/numTP
        Rmax = max(Rdev)
        Rmax2 = max(Rdev2)

        print("One Step Estimator Average Error:",Ravg2,"Max error:",Rmax2,"Time:",onestep)        
        print("Two Step Estimator Average Error:",Ravg,"Max error:",Rmax,"Time:",twostep)        

        traces2 = []
        traces.append(cond)
        traces2.append(cond2)
        #print("RKHS Sigma = ",sigma[i],"Average Error:",SigmaErrors[i],"Max error:",MaxErrors[i],"Time:",times[i])        
        # print(string,"Average Error:",SigmaErrors[s],"Max error:",MaxErrors[s],"Time:",times[s])        
        #plt.plot(testPoints,traces[i], label = 'RKHS curve sigma = '+ str(sigma[i]))
        plt.plot(testPoints,cond, label = "Two Step")
        plt.plot(testPoints,cond2, label = "One Step")

        cond = []




    print("Prob.py Time:", end1-start1)
    print("Prob.py Average Error:",Pavg,"Max error:",max(Pdev))

    
        
    plt.plot(testPoints,sq, label = 'Ideal Curve')
    plt.plot(testPoints,probpy, label = 'ProbSpace Y|X')    
    plt.legend()
    plt.show()


    

    
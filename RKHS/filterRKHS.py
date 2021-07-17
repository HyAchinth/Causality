import sys

import numpy
sys.path.append('.')
sys.path.append('../')
from Probability.Prob import *
from synth import getData
from numpy import *
from RKHSmod.rkhs import RKHS
from Condtest2 import *
import time
from math import *

def ref(x,choice):
    if choice == 1:
        return math.tanh(x) + math.sin(conditional)
    else:
        return math.sin(x) + math.tanh(conditional)

def kcdf(x1, x2=0, kparms=[]):
    # CDF Kernel
    sigma = kparms[0]
    return (1 + erf(abs(x2-x1)/sigma*sqrt(2))) / 2   


path = '../models/Cprobdata2.csv'
d = getData.DataReader(path)
data = d.read()

P = ProbSpace(data)

#conditional X value
conditional = 3

#choice 1=X(tanh) , 0=Y(sinh) (choose 1 to condition on X, 0 for Y)
choice = 1

rtime1 = time.time()
if choice == 1:
    FilterData, parentProb, finalQuery = P.filter([('X',conditional)])
else:
    FilterData, parentProb, finalQuery = P.filter([('Y',conditional)])

X = FilterData['X']
Y = FilterData['Y']
Z = FilterData['Z']


filterlen = len(Z)

s = 0.2 #sigma

if choice == 0:
    r1 = RKHS(X,kparms=[s])
else:
    r1 = RKHS(Y,kparms=[s])

r2 = RKHS(Z,kparms=[s])
rtime2 = time.time()
Rtime = rtime2 - rtime1

Ptime = 0

testPoints = []
if choice == 1:
    testMin = -10
    testMax = 10
else:
    testMin = -6
    testMax = 6
tp = testMin
numTP = 200
interval = (testMax - testMin) / numTP

Rdev = []
Pdev = []
cond = []
ideal = []
Probpy = []
RSqerr = []

PSqerr = []
Sqmean = []

for i in range(numTP + 1):
    testPoints.append(tp)   
    #ans = evalYx(tp,r1,r2,choice[j],e)
    t1 = time.time()
    ans = m(tp,r1,r2)
    if ans != 0:
        cond.append(ans)
    elif i== 0:
        cond.append(0)
    else:
        cond.append(cond[i-1])
    t2 = time.time()
    Rtime =Rtime + (t2-t1)
    
    t1 = time.time()
    if choice ==0:
        p = P.distr('Z', [('X', tp), ('Y', conditional)]).E()
    else:
        p = P.distr('Z', [('Y', tp), ('X', conditional)]).E()
    Probpy.append(p)
    t2 = time.time()
    Ptime = Ptime + (t2-t1)
         
    r = ref(tp,choice)
    ideal.append(r)            
    err = abs(r-ans)
    Rdev.append(err)
    err2 = abs(r-p)
    Pdev.append(err2)   
    tp += interval

ymean = sum(ideal)/len(ideal)

for i in range(numTP + 1):
    RSqerr.append(Rdev[i]**2)
    PSqerr.append(Pdev[i]**2)
    Sqmean.append((ideal[i]-ymean)**2)

R2rkhs = 1 - sum(RSqerr)/sum(Sqmean)
R2probpy = 1 - sum(PSqerr)/sum(Sqmean)


print("Filtered Datasize:",filterlen)
print("Filter-RKHS\tAverage Error:",sum(Rdev)/len(Rdev),"\tR2 error:",R2rkhs,"\ttime:",Rtime)
print("ProbSpace \tAverage Error:",sum(Pdev)/len(Pdev),"\tR2 error:",R2probpy,"\ttime:",Ptime)

if choice == 0:
    labelstr = 'Filter-RKHS Z|X,Y='+str(conditional)
else:
    labelstr = 'Filter-RKHS Z|Y,X='+str(conditional)

plt.plot(testPoints,ideal, label = 'Ideal Curve',color='#000000')
plt.plot(testPoints,Probpy, label = 'ProbSPace')
plt.plot(testPoints,cond, label = labelstr)    
plt.legend()
plt.show()


